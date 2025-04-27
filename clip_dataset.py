import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import pickle
import os
import sys
from typing import Tuple, List


class CocoClipDataset(Dataset):
    def __init__(self, dataset_path: str, prefix_length: int, gpt2_variant: str = "gpt2", should_normalize_prefix: bool = False):
        self.prefix_length = prefix_length
        self.normalize_clip = should_normalize_prefix
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_variant)
        self._load_data(dataset_path)

    def _load_data(self, dataset_path: str):
        with open(dataset_path, 'rb') as f:
            stored_data = pickle.load(f)
        print(f"Dataset contains {len(stored_data['clip_embedding'])} samples.")
        sys.stdout.flush()

        self.clip_features = stored_data["clip_embedding"]
        caption_metadata = stored_data["captions"]
        self.image_keys = [item["image_id"] for item in caption_metadata]
        self.caption_texts = [item["caption"] for item in caption_metadata]

        tokens_file = f"{dataset_path[:-4]}_tokens.pkl"
        if os.path.isfile(tokens_file):
            with open(tokens_file, 'rb') as f:
                self.processed_tokens, self.caption_to_clip, self.max_seq_length = pickle.load(f)
        else:
            self._process_and_cache_tokens(tokens_file, caption_metadata)

        caption_lengths = torch.tensor([len(self.processed_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_length = min(int(caption_lengths.mean() + caption_lengths.std() * 10),
                                  int(caption_lengths.max()))

    def _process_and_cache_tokens(self, save_path: str, caption_data: List[dict]):
        self.processed_tokens = []
        self.caption_to_clip = []
        longest = 0

        for entry in caption_data:
            encoded = torch.tensor(self.tokenizer.encode(entry['caption']), dtype=torch.int64)
            self.processed_tokens.append(encoded)
            self.caption_to_clip.append(entry["clip_embedding"])
            longest = max(longest, encoded.shape[0])

        with open(save_path, 'wb') as f:
            pickle.dump([self.processed_tokens, self.caption_to_clip, longest], f)

    def __len__(self) -> int:
        return len(self.processed_tokens)

    def _prepare_token(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        token_seq = self.processed_tokens[idx]
        pad_needed = self.max_seq_length - token_seq.size(0)

        if pad_needed > 0:
            token_seq = torch.cat((token_seq, torch.full((pad_needed,), -1, dtype=torch.int64)))
        elif pad_needed < 0:
            token_seq = token_seq[:self.max_seq_length]

        valid_mask = token_seq.ge(0)
        token_seq[~valid_mask] = 0
        mask = valid_mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)

        return token_seq, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_seq, mask = self._prepare_token(idx)
        clip_embedding = self.clip_features[self.caption_to_clip[idx]]

        if self.normalize_clip:
            clip_embedding = clip_embedding.float()
            clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)

        return token_seq, mask, clip_embedding

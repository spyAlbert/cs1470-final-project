import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from typing import Optional
from mlp import MLP
from transformer import TransformerMapper

class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, clip_len: Optional[int] = None, prefix_dim: int = 512,
                 transformer_layers: int = 8, projection_method: str = 'mlp'):
        super().__init__()
        self.prefix_length = prefix_length
        self._initialize_gpt()
        self._initialize_projection(prefix_dim, projection_method, clip_len, transformer_layers)

    def _initialize_gpt(self):
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_hidden_size = self.gpt_model.transformer.wte.embedding_dim

    def _initialize_projection(self, prefix_dim, projection_method, clip_len, transformer_layers):
        if projection_method == 'mlp':
            self.prefix_mapper = MLP((prefix_dim,
                                      (self.prefix_length * self.gpt_hidden_size) // 2,
                                      self.prefix_length * self.gpt_hidden_size))
        else:
            self.prefix_mapper = TransformerMapper(prefix_dim, self.gpt_hidden_size, self.prefix_length,
                                                   clip_len, transformer_layers)

    def create_dummy_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.prefix_length), dtype=torch.long, device=device)

    def forward(self, token_sequence: torch.Tensor, clip_prefix: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None, target_sequence: Optional[torch.Tensor] = None):
        prefix_embeds = self.prefix_mapper(clip_prefix).view(-1, self.prefix_length, self.gpt_hidden_size)
        token_embeds = self.gpt_model.transformer.wte(token_sequence)
        full_embedding = torch.cat((prefix_embeds, token_embeds), dim=1)

        if target_sequence is not None:
            dummy_targets = self.create_dummy_tokens(token_sequence.size(0), token_sequence.device)
            target_sequence = torch.cat((dummy_targets, token_sequence), dim=1)

        output = self.gpt_model(inputs_embeds=full_embedding, labels=target_sequence, attention_mask=attention_mask)
        return output

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return super().parameters(recurse=False) if not recurse else self.prefix_mapper.parameters()

    def set_training_mode(self, is_training: bool = True):
        self.train(is_training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.gpt_model.eval()
        return self
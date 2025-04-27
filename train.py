import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from clip_dataset import CocoClipDataset
from clip_caption import ClipCaptionModel, ClipCaptionPrefix

def setup_training(model, dataset, args, learning_rate=2e-5, warmup_steps=5000):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    data_loader = DataLoader(
        dataset, batch_size=args.bs, shuffle=True, drop_last=True
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.epochs * len(data_loader)
    )
    return model, data_loader, optimizer, scheduler, device

def train_one_epoch(model, loader, optimizer, scheduler, device, save_dir, prefix, dataset, epoch_idx, save_every, args):
    model.train()
    progress = tqdm(loader, desc=f"{prefix} Epoch {epoch_idx}")
    
    for batch_idx, (input_ids, attention_mask, clip_features) in enumerate(progress):
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        clip_features = clip_features.to(device).float()

        output = model(input_ids, clip_features, attention_mask)
        logits = output.logits[:, dataset.prefix_length - 1 : -1]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=0
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress.set_postfix(loss=loss.item())

        if (batch_idx + 1) % 10000 == 0:
            checkpoint_path = os.path.join(save_dir, f"{prefix}_latest.pt")
            torch.save(model.state_dict(), checkpoint_path)
    
    progress.close()

    if (epoch_idx % save_every == 0) or (epoch_idx == args.epochs - 1):
        final_path = os.path.join(save_dir, f"{prefix}-{epoch_idx:03d}.pt")
        torch.save(model.state_dict(), final_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_ViT-B_32_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', action='store_true')
    parser.add_argument('--mapping_type', default='mlp')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', action='store_true')
    args = parser.parse_args()

    dataset = CocoClipDataset(
        args.data, args.prefix_length, should_normalize_prefix=args.normalize_prefix
    )
    clip_dim = 640 if args.is_rn else 512

    if args.only_prefix:
        print("Training prefix only...")
        model = ClipCaptionPrefix(
            prefix_length=args.prefix_length,
            clip_len=args.prefix_length_clip,
            prefix_dim=clip_dim,
            transformer_layers=args.num_layers,
            projection_method=args.mapping_type
        )
    else:
        print("Training full model with prefix and GPT...")
        model = ClipCaptionModel(
            prefix_length=args.prefix_length,
            clip_len=args.prefix_length_clip,
            prefix_dim=clip_dim,
            transformer_layers=args.num_layers,
            projection_method=args.mapping_type
        )
    sys.stdout.flush()

    model, loader, optimizer, scheduler, device = setup_training(model, dataset, args)

    for epoch in range(args.epochs):
        print(f"\n=== Starting epoch {epoch} ===")
        train_one_epoch(
            model, loader, optimizer, scheduler, device,
            save_dir=args.out_dir,
            prefix=args.prefix,
            dataset=dataset,
            epoch_idx=epoch,
            save_every=args.save_every,
            args = args
        )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import json
import pickle
import argparse
from pathlib import Path

import torch
print(torch.cuda.is_available()) 
import clip
from PIL import Image
from skimage import io as skio
from tqdm import tqdm


def load_json(path: Path):
    """Read caption json to memory."""
    with path.open("r") as fp:
        captions = json.load(fp)
    print(f"{len(captions)} captions loaded from json")
    return captions


def find_image(img_id: int) -> Path:
    """
    Given a COCO image id, locate the JPG file in either
    train2014 or val2014 folder.
    """
    try:
        img_id = int(img_id)  # Try to convert img_id to integer
    except ValueError:
        raise ValueError(f"Image id {img_id} is not a valid integer.")
    filename = f"COCO_train2014_{img_id:012d}.jpg"
    train_path = Path("./data/coco/train2014") / filename
    if train_path.is_file():
        return train_path

    filename = f"COCO_val2014_{img_id:012d}.jpg"
    val_path = Path("./data/coco/val2014") / filename
    if val_path.is_file():
        return val_path

    raise FileNotFoundError(f"Image {img_id} not found in train/val splits.")


def extract_embeddings(model, preprocess, captions, device) -> tuple[list[torch.Tensor], list[dict]]:
    """
    Iterate over captions, load corresponding images,
    and compute CLIP image embeddings.
    """
    features, meta = [], []

    for idx in tqdm(range(len(captions))):
        item = captions[idx]
        img_path = find_image(item["image_id"])

        image_np = skio.imread(img_path)
        img_tensor = preprocess(Image.fromarray(image_np)).unsqueeze(0).to(device)

        with torch.inference_mode():
            emb = model.encode_image(img_tensor).cpu()

        # store tiny index to keep alignment without bloating JSON
        item["clip_embedding"] = idx
        features.append(emb)
        meta.append(item)

        # checkpoint every 10k samples
        if (idx + 1) % 10_000 == 0:
            yield idx + 1, features, meta
            features, meta = [], []

    # final remainder
    yield len(captions), features, meta


def dump_checkpoint(out_path: Path, feats: list[torch.Tensor], infos: list[dict]):
    """Serialize current batch to a pkl file (append / overwrite)."""
    tensor_cat = torch.cat(feats, dim=0)
    payload = {"clip_embedding": tensor_cat, "captions": infos}
    with out_path.open("wb") as fp:
        pickle.dump(payload, fp)


def run(clip_model_type: str):
    # ----------- setup -----------
    device = torch.device("cpu")
    model_id = clip_model_type.replace("/", "_")
    output_pkl = Path(f"./data/coco/oscar_split_{model_id}_train.pkl")

    model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    caption_file = Path("./data/coco/annotations/train_caption.json")
    captions = load_json(caption_file)

    # ----------- main loop -----------
    progress = 0
    for progress, feats, infos in extract_embeddings(model, preprocess, captions, device):
        dump_checkpoint(output_pkl, feats, infos)

    print("Done")
    print(f"{progress} embeddings saved to {output_pkl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP image embeddings for COCO captions.")
    parser.add_argument(
        "--clip_model_type",
        default="ViT-B/32",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"),
        help="Which CLIP backbone to load.",
    )
    args = parser.parse_args()
    run(args.clip_model_type)

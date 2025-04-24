# create 1/10 random sampled dataset
import os
import json
import random
import shutil
from pathlib import Path
from skimage import io as skio
from PIL import Image

def load_json(path: Path):
    """Load captions JSON file into memory."""
    with path.open("r") as fp:
        captions = json.load(fp)
    print(f"{len(captions)} captions loaded from json")
    return captions

def sample_dataset(captions, fraction=0.1):
    """Randomly sample a fraction of the dataset."""
    sample_size = int(len(captions) * fraction)
    sampled_captions = random.sample(captions, sample_size)
    print(f"Sampled {sample_size} captions from the dataset.")
    return sampled_captions

def create_new_dirs(base_path: Path):
    """Create directories for the new dataset."""
    os.makedirs(base_path / "train2014", exist_ok=True)
    os.makedirs(base_path / "val2014", exist_ok=True)
    os.makedirs(base_path / "annotations", exist_ok=True)
    print(f"Created new directories at {base_path}")

def find_image(img_id: int, train_path: Path, val_path: Path) -> Path:
    """Find image in train2014 or val2014 based on image_id."""
    try:
        img_id = int(img_id)  # Try to convert img_id to integer
    except ValueError:
        raise ValueError(f"Image id {img_id} is not a valid integer.")
    filename = f"COCO_train2014_{img_id:012d}.jpg"
    train_image_path = train_path / filename
    if train_image_path.is_file():
        return train_image_path
    
    filename = f"COCO_val2014_{img_id:012d}.jpg"
    val_image_path = val_path / filename
    if val_image_path.is_file():
        return val_image_path

    raise FileNotFoundError(f"Image {img_id} not found in train/val splits.")

def copy_image(img_path: Path, new_dir: Path):
    """Copy image to the new directory."""
    shutil.copy(img_path, new_dir)

def preprocess_and_copy_dataset(captions, output_dir: Path, train_path: Path, val_path: Path):
    """Preprocess the dataset: sample 1/10th, copy images, and store captions."""
    sampled_captions = sample_dataset(captions, fraction=0.1)
    create_new_dirs(output_dir)

    # Define paths for the new directories
    new_train_dir = output_dir / "train2014"
    new_val_dir = output_dir / "val2014"
    
    # Create a new list for the sampled captions
    new_captions = []
    
    # Process the sampled captions
    for item in sampled_captions:
        img_id = item["image_id"]
        try:
            img_path = find_image(img_id, train_path, val_path)
        except FileNotFoundError:
            continue
        
        # Determine the new directory (train or val) and copy the image
        if "train" in str(img_path):
            new_img_dir = new_train_dir
        else:
            new_img_dir = new_val_dir
        
        # Copy the image to the new directory
        copy_image(img_path, new_img_dir)
        
        # Add the item to the new caption list
        new_captions.append(item)
    
    # Save the new captions JSON to the annotations directory
    annotations_path = output_dir / "annotations" / "train_caption.json"
    with annotations_path.open("w") as fp:
        json.dump(new_captions, fp)
    print(f"Saved {len(new_captions)} sampled captions to {annotations_path}")

def main():
    # Define paths to the original dataset
    original_train_path = Path("./data/coco/train2014")
    original_val_path = Path("./data/coco/val2014")
    
    # Define output directory for the new dataset
    output_dir = Path("./data/coco_sampled")
    
    # Load the captions file
    caption_file = Path("./data/coco/annotations/train_caption.json")
    captions = load_json(caption_file)
    
    # Process the dataset: sample, copy images, and store captions
    preprocess_and_copy_dataset(captions, output_dir, original_train_path, original_val_path)
    
    print("Preprocessing and dataset reduction complete!")

if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import json
from bleu import compute_bleu
import re

def extract_image_ids(data_folder, image_ids, mlp_map, transformer_map):

    for filename in os.listdir(data_folder):
        if filename.endswith(".jpg"):  
            image_id = filename.replace("COCO_val2014_", "").replace(".jpg", "")
            image_ids.add(int(image_id)) 

            image_path = os.path.join(data_folder, filename)
            
            command1 = [
                "python", "predict.py",
                "--mapping_type", 'mlp',  
                "--image", image_path 
            ]
            
            result = subprocess.run(command1, capture_output=True, text=True)
            
            if result.returncode == 0:
                caption1 = result.stdout.strip()  
                print(f"Caption for {filename}: {caption1}")
                mlp_map[image_id] = caption1
            else:
                print(f"Error generating caption for {filename}: {result.stderr.strip()}")
            command2 = [
                "python", "predict.py",  
                "--mapping_type", 'transformer', 
                "--image", image_path 
            ]
            result = subprocess.run(command2, capture_output=True, text=True)
            
        
            if result.returncode == 0:
                caption2 = result.stdout.strip() 
                print(f"Caption for {filename}: {caption2}")
                transformer_map[image_id] = caption2
            else:
                print(f"Error generating caption for {filename}: {result.stderr.strip()}")
    return image_ids

def get_real_captions(image_ids, real_map, path_caption):
    with open(path_caption, 'r') as f:
        data = json.load(f)
    for item in data:
        image_id = item['image_id']
        image_id = int(image_id)
        if image_id in image_ids:
            caption = item['caption']
            print(f"Image ID: {image_id}, Caption: {caption}")
            real_map.setdefault(image_id, []).append(caption)
    
def extract_captions(mlp_map, transformer_map, image_ids):
    file_path = "predict.txt"
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        match = re.match(r"Caption for (COCO_val2014_\d+).jpg: (.+)", line.strip())
        if match:
            image_id_str, caption = match.groups()
            image_id = int(image_id_str.replace("COCO_val2014_", "").replace(".jpg", ""))
            image_ids.add(image_id) 
            
            if i % 2 == 0:  
                mlp_map[image_id] = caption
            else: 
                transformer_map[image_id] = caption

    return mlp_map, transformer_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_caption', default="./data/coco/annotations/train_caption.json")
    parser.add_argument('--path_images', default="./data/coco/bleu")
    args = parser.parse_args()
    image_ids = set() 
    mlp_map = {}
    transformer_map = {}
    extract_image_ids(args.path_images, image_ids, mlp_map, transformer_map)
    #extract_captions(mlp_map, transformer_map, image_ids)
    print(mlp_map)
    print(transformer_map)
    with open('mlp_map.json', 'w') as json_file:
        json.dump(mlp_map, json_file, indent=4)
    with open('transformer_map.json', 'w') as json_file:
        json.dump(transformer_map, json_file, indent=4)
    real_map = {}
    get_real_captions(image_ids, real_map, args.path_caption)
    with open('real_map.json', 'w') as json_file:
        json.dump(real_map, json_file, indent=4)
    score_mlp = 0
    score_transformer = 0
    count = 0
    for id in image_ids:
        if real_map.get(id) == None:
            continue
        count += 1
        real_captions = real_map[id]
        mlp_caption = mlp_map[id]
        transformer_caption = transformer_map[id]
        score_mlp += compute_bleu(mlp_caption, real_captions)
        score_transformer += compute_bleu(transformer_caption, real_captions)
    #get average
    score_mlp /= count
    score_transformer /= count
    print(f"Count: {count}")
    print(f"Score for mlp: {score_mlp}")
    print(f"Score for transformer: {score_transformer}")

if __name__ == "__main__":
    main()

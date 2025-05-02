# Image Captioning via CLIP Prefix Tuning

## Overview

This project explores how powerful pre-trained vision and language models, such as CLIP and GPT-2, can be leveraged for image captioning without fine-tuning the large models themselves. Our approach reimplements the "ClipCap: CLIP Prefix for Image Captioning" model, proposed by Ron et al. We aim to bridge the vision and language domains by using a lightweight Transformer (or MLP) to map visual features from CLIP into GPT-2's language embedding space.

## Introduction

Our project focuses on efficiently using CLIP and GPT-2 for image captioning tasks, by training a small mappting network (mlp/transformer) that generates prefix tokens for GPT-2 based on CLIP’s visual embeddings. The aim is to generate grammatically coherent and semantically relevant captions without requiring fine-tuning of large models like CLIP and GPT-2.

## Related Work

- **ClipCap:CLIP Prefix for Image Captioning** by Ron et al.

## Data

We use a subset of the **COCO Captions Dataset**, which contains 56,674 captions in total: 33,841 images for training and 12,505 images for validation. The dataset is preprocessed by resizing and normalizing images for CLIP, and captions are tokenized using GPT-2's tokenizer. You can download the dataset [here](https://drive.google.com/file/d/1sAsUIo_W36DFZ6fXBRRkWGGHTIqA-LRx/view).

## Methodology

Our pipeline consists of:

1. **CLIP** for extracting image embeddings.
2. **A Transformer (or MLP)** that maps CLIP embeddings to prefix tokens for GPT-2.
3. **GPT-2** which generates captions conditioned on the prefix tokens.

### Train

1. Run `python clip_preprocess.py` to create CLIP embeddings, or download [here](https://drive.google.com/file/d/13qzq6dw6aYx79sy1SrJ2whvPG2YP9v1a/view?usp=sharing).
2. Run `python train.py` to train the MLP mapper and fine-tune GPT-2.
3. Run `python train.py --only_prefix --mapping_type transformer --prefix_length 40 --prefix_length_clip 40` to train the Transformer mapper with a frozen GPT-2.

We also implemented a **TensorFlow version** of CLIP. You can run `clip_preprocess_tf.py` and `train_tf.py` for that version.

### Predict

Run `python predict.py --mapping_type mlp/transformer --image image_path --model model_weight_path` to predict caption.

## Metrics

We evaluate the model using the following image captioning metrics:

- **BLEU-4**: Measures n-gram overlap with ground truth captions.

Run `python cal_bleu.py --path_caption path_to_image_captions --path_images path_to_images_folder` to calculate the average BLEU-4 score for both MLP and Transformer mapping networks. Image names should match the format `COCO_val2014_XXX`, where `XXX` is the image ID.

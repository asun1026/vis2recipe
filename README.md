# Visual-To-Recipe: Automated Recipe Generation from Food Images

This repository contains the implementation of our paper "Visual-To-Recipe: Automated Recipe Generation from Food Images," which presents a system for generating detailed cooking recipes directly from food dish images.

## Project Overview

We introduce a deep learning system that bridges the gap between visual understanding and procedural text generation for cooking recipes. Our approach employs a lightweight adaptation strategy that connects powerful pre-trained unimodal models:

1. A frozen vision encoder extracts rich visual features from food images
2. A frozen large language model (LLM) generates coherent recipe text
3. A minimal trainable projection layer connects these components

This approach allows us to leverage extensive knowledge embedded in pre-trained models while requiring significantly fewer computational resources compared to end-to-end fine-tuning.

## Key Components

### Frozen Recipe1M Model (`models/recipe1m/`)

The `recipe1m` directory contains the implementation of the frozen im2recipe model used as our vision encoder:

- **model.py**: Implements the encoder that extracts embeddings from food images
- **experiments.ipynb**: Contains runnable cells that prepare data and train model, and provide demo examples

We use the official pre-trained im2recipe model checkpoint (`model_e500_v-8.950.pth.tar`), which remains frozen during our training process.

### Lightweight Alternative Method (`train_recipe_layers.py`)

This script implements our alternative lightweight vision-language integration approach:

- Connects a frozen EfficientNet-B0 image encoder to a frozen LLM (DeepSeek-R1-Distill-Llama-8B)
- Implements the trainable projection layer that maps 1280-dimensional image embeddings to the LLM's 4096-dimensional space
- Adds the projected image embedding to the first token of the input prompt
- Trains only the projection layer parameters using standard autoregressive cross-entropy loss

### Visualization Utilities (`vis.py`)

The `vis.py` module provides tools for visualizing the model outputs and results

## Dataset

We utilize the Recipe1M dataset, which contains over one million cooking recipes paired with images. Due to computational constraints, we work with a pre-processed subsets of the data.

## Requirements

A complete list of dependencies is available in `requirements.txt`.

## Acknowledgments

We used Google Gemini Pro 2.5 to assist in editing and style of our paper. We build upon the im2recipe model from Salvador et al. and utilize the Recipe1M dataset in our work.


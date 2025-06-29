# Face Verification Using Siamese Network with EfficientNetB0

![Project Banner](https://via.placeholder.com/800x200?text=Face+Verification+Siamese+Network)  
*A lightweight yet accurate face verification system for real-world applications.*

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Project Overview
This project implements a **Siamese Neural Network** with **EfficientNetB0** backbone for face verification tasks. It compares pairs of facial images to determine if they belong to the same person, achieving **87% accuracy** while being optimized for edge devices.

## Key Features
- 🎭 **Siamese Architecture**: Twin networks with shared weights
- ⚡ **EfficientNetB0**: Lightweight pre-trained feature extractor
- 📏 **L1 Distance Metric**: Computes similarity between embeddings
- 📱 **Edge-Optimized**: 4x fewer parameters than baseline CNN
- 🔄 **Contrastive Learning**: Works with limited labeled data

## Installation
```bash
```bash
   git clone https://github.com/ganesh2427/recognition_rawmodel.git
   cd recognition_rawmodel
   ```
   
Requirements
```bash
The required dependencies are listed in the `requirements.txt` file. To install them, run:

pip install -r requirements.txt
 

## Usage

### 1. Prepare Dataset
Structure your dataset as:
dataset/
├── anchor/ # Reference images
├── positive/ # Same-person variants
└── negative/ # Different-person images

Copy

### 2. Train the Model
```python
from model import SiameseNetwork

model = SiameseNetwork(backbone='efficientnetb0')
model.train(data_dir='dataset/', epochs=50)


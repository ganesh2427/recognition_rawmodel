# Face Verification with Siamese Network & EfficientNetB0

## Project Description

This project provides an face verification system using a Siamese Neural Network as the backbone, enhanced with a pre-trained EfficientNetB0 feature extractor. It is designed to compare pairs of facial images and determine if they belong to the same person, making it suitable for authentication and security applications. The model achieves high accuracy while remaining lightweight, making it ideal for deployment on edge devices. The repository includes all necessary code, instructions, and a sample data structure to help you get started quickly.

### This repository contains:

1. Model Architecture: Siamese Neural Network with EfficientNetB0 as the feature extractor.
2. Training Scripts: Code for training and evaluating the face verification model.
3. Data Structure: Example folder structure for organizing anchor, positive, and negative images.
4. Preprocessing: Utilities for image loading, augmentation, and preprocessing.
5. Requirements: requirements.txt file listing all Python dependencies.
6. Documentation: Step-by-step instructions for installation, usage, and customization.
7. Sample Results: Example outputs and accuracy metrics.
8. License: Open-source license for academic and commercial use.




## Table of Contents
- [Background](#Background)
- [Install](#Install)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Background
This project implements a **Siamese Neural Network** with **EfficientNetB0** backbone for face verification tasks. It compares pairs of facial images to determine if they belong to the same person, achieving **87% accuracy** while being optimized for edge devices.

## Key Features
- 🎭 **Siamese Architecture**: Twin networks with shared weights
- ⚡ **EfficientNetB0**: Lightweight pre-trained feature extractor
- 📏 **L1 Distance Metric**: Computes similarity between embeddings
- 📱 **Edge-Optimized**: 4x fewer parameters than baseline CNN
- 🔄 **Contrastive Learning**: Works with limited labeled data

## Installation

### Prerequisites

- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **pip**: Comes with Python, but ensure it's up to date (`python -m pip install --upgrade pip`)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ganesh2427/recognition_rawmodel.git
   cd recognition_rawmodel
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

Now you are ready to use the project!

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


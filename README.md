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


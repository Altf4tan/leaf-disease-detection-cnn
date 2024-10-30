# Leaf Disease Detection using CNN

## Overview
This project aims to classify plant leaf images to detect and identify diseases using a Convolutional Neural Network (CNN). The model is trained on the PlantVillage dataset, which consists of images labeled by different disease categories. This tool can assist in the early detection of plant diseases, promoting better management and treatment.

## Features
- Preprocessing of images (resizing, normalization)
- Train/validation/test split for model evaluation
- CNN architecture for image classification
- Model checkpointing to save the best model based on validation accuracy

## Dataset
- **Source**: [PlantVillage dataset](https://example-link-to-dataset.com)
- **Description**: A collection of color images categorized by disease type.

## Model Architecture
The model consists of several convolutional and pooling layers, batch normalization, and dense layers for classification. The architecture is optimized to achieve high accuracy on the given dataset.

## Getting Started

### Prerequisites
- Python 3.x
- OpenCV
- TensorFlow
- Keras
- NumPy
- scikit-learn

### Installation
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/username/leaf-disease-detection-cnn.git
cd leaf-disease-detection-cnn
pip install -r requirements.txt

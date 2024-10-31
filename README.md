

# Plant Health Assessment: Disease Classification and Color Segmentation

## Overview
This project provides a comprehensive solution for assessing plant health through image analysis. It includes two main tools: a CNN-based classifier to identify diseases in leaf images and a color segmentation tool to analyze green and brown areas, which helps indicate healthy and diseased regions.

## Features
- **Disease Classification**: A Convolutional Neural Network (CNN) classifies leaf images into different disease categories.
- **Color Segmentation**: HSV color filtering isolates green (healthy) and brown (diseased) regions in leaf images for quick visual assessment.

## Dataset
- **Source**: [PlantVillage Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)
- **Description**: Contains labeled leaf images for various plant diseases.

## Model and Tools

### 1. Disease Classification with CNN
   - The CNN model consists of convolutional, pooling, and dense layers optimized for high accuracy on the PlantVillage dataset.
   - **Model Checkpointing**: Saves the best model based on validation accuracy for robust performance.

### 2. Leaf Color Segmentation
   - HSV color segmentation highlights green and brown areas in leaf images.
   - Displays original and segmented images for easy analysis of healthy versus potentially diseased areas.

## Getting Started

### Prerequisites
- Python 3.x
- OpenCV
- TensorFlow
- Keras
- NumPy
- scikit-learn

### Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/username/plant-health-assessment.git
cd plant-health-assessment
pip install -r requirements.txt
```

### Usage

1. **Disease Classification**:
    - Place the dataset in the specified directory (`data_directory` in the code).
    - Run the training script:
      ```bash
      python train_model.py
      ```

2. **Color Segmentation**:
    - Add the target image (e.g., `leaf2.jpg`) to the main directory.
    - Run the segmentation script:
      ```bash
      python segment_color.py
      ```

### Results
- The CNN model achieves a test accuracy of approximately `56%`, with the best model saved as `model_45.h5`.
- The color segmentation tool displays the original and segmented images highlighting green and brown regions.

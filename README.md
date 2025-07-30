## Emotion Detection with Activation Maps Visualization

This project implements a Convolutional Neural Network (CNN) for emotion detection and visualizes activation maps to understand which image regions activate different filters in the network.

## Project Overview

The project consists of several components:
1. **Data Preparation**: Loading and preprocessing emotion images
2. **Model Training**: Building and training a CNN for emotion classification
3. **Model Evaluation**: Assessing model performance with metrics and visualizations
4. **Activation Maps Visualization**: Understanding what the CNN "sees" in images


 **Data Organization**:
   data is organized as follows:
   ```
   train:test for emotion model/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprise/
   ```


## Model Architecture

The CNN consists of:
- **3 Convolutional Blocks**: Each with Conv2D, BatchNormalization, MaxPooling, and Dropout
- **Dense Layers**: 512 and 256 neurons with dropout for regularization
- **Output Layer**: 7 neurons (one for each emotion class)

## Key Features

### Activation Maps Visualization
- **Filter Visualization**: Shows what each convolutional filter detects
- **Overlay Maps**: Combines activation maps with original images
- **Multi-layer Analysis**: Visualizes activations from different layers

### Model Evaluation
- **Confusion Matrix**: Shows class-wise prediction accuracy
- **Classification Report**: Precision, recall, and F1-score for each emotion


## Technical Details

### Data Preprocessing
- Images resized to 48x48 pixels (grayscale)
- Pixel values normalized to [0, 1]
- Data augmentation applied to training data (rotation, shift, zoom, flip)

### Model Training
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Categorical crossentropy
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction

## Notes

- The model requires at least 70% accuracy as per project requirements
- Activation maps help understand model interpretability
- All visualizations are saved in high resolution (300 DPI)
- The project follows best practices for deep learning project organization




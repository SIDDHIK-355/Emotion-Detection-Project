## Emotion Detection with Activation Maps Visualization

This project implements a Convolutional Neural Network (CNN) for emotion detection and visualizes activation maps to understand which image regions activate different filters in the network.

## Project Overview

The project consists of several components:
1. **Data Preparation**: Loading and preprocessing emotion images
2. **Model Training**: Building and training a CNN for emotion classification
3. **Model Evaluation**: Assessing model performance with metrics and visualizations
4. **Activation Maps Visualization**: Understanding what the CNN "sees" in images

## Project Structure

```
emotion_activation_maps_project/
├── src/
│   ├── data_preparation.py      # Data loading and preprocessing
│   ├── model_training.py        # CNN model building and training
│   ├── model_evaluation.py      # Model evaluation and metrics
│   └── activation_maps.py       # Activation maps visualization
├── models/                      # Saved model files
├── activation_maps_output/      # Generated visualizations
├── notebooks/                   # Jupyter notebooks
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Organization**:
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



### 2. Model Training
```bash
python model_training.py
```
This will:
- Build a CNN with multiple convolutional layers
- Train the model on your emotion dataset
- Save the best model to `models/best_emotion_model.h5`

### 3. Model Evaluation
```bash
python model_evaluation.py
```
This will:
- Load the trained model
- Evaluate performance on test data
- Generate confusion matrix and classification report
- Save visualizations to `activation_maps_output/`

### 4. Activation Maps Visualization
```bash
python activation_maps.py
```
This will:
- Load the trained model
- Extract activation maps from convolutional layers
- Visualize which image regions activate different filters
- Create overlays showing activation patterns

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
/
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


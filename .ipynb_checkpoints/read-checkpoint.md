# Emotion Detection with Activation Maps Visualization

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
   Ensure your data is organized as follows:
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

## Usage

### 1. Data Preparation
```bash
cd src
python data_preparation.py
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

### Model Evaluation
- **Confusion Matrix**: Shows class-wise prediction accuracy
- **Classification Report**: Precision, recall, and F1-score for each emotion
- **Training History**: Plots of accuracy and loss over training epochs

## Output Files

After running the complete pipeline, you'll find:
- `models/best_emotion_model.h5`: Trained model with best validation accuracy
- `activation_maps_output/confusion_matrix.png`: Model performance visualization
- `activation_maps_output/training_history.png`: Training progress plots
- `activation_maps_output/*_activations.png`: Activation maps for different layers
- `activation_maps_output/*_overlay.png`: Activation maps overlaid on original images

## Technical Details

### Data Preprocessing
- Images resized to 48x48 pixels (grayscale)
- Pixel values normalized to [0, 1]
- Data augmentation applied to training data (rotation, shift, zoom, flip)

### Model Training
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Categorical crossentropy
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction
- **Batch Size**: 32
- **Epochs**: Up to 50 (with early stopping)

### Activation Maps
- Extracted from convolutional layers
- Normalized for visualization
- Overlaid on original images with transparency

## Requirements

- Python 3.7+
- TensorFlow 2.8+
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

## Notes

- The model requires at least 70% accuracy as per project requirements
- Activation maps help understand model interpretability
- All visualizations are saved in high resolution (300 DPI)
- The project follows best practices for deep learning project organization

## Troubleshooting

1. **Memory Issues**: Reduce batch size in `data_preparation.py`
2. **Training Time**: Reduce number of epochs or use GPU acceleration
3. **Data Path Issues**: Ensure correct path to your train/test data folders

## Contributing

Feel free to modify the model architecture, add new visualization techniques, or improve the evaluation metrics. 
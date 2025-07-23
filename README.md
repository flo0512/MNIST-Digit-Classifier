# MNIST Digit Classifier

This is my first neural network project using TensorFlow and Keras.  
The model learns to recognize handwritten digits from the MNIST dataset.

## What I Learned

- Basics of neural networks (layers, neurons, activation functions)
- How to train and evaluate a model with Keras
- Data normalization and model accuracy evaluation
- Overfitting and generalization
- Dropout regularization and its effect on model performance

## Model Architecture

### Base Model:

- Flatten layer (input shape: 28x28)
- Dense layer with 128 neurons and ReLU activation
- Output layer with 10 neurons and softmax activation

### Overfit Model:

- Same as base, trained with 100 epochs to observe overfitting behavior

### Dropout Model:

- Dense layer with 128 neurons and ReLU activation
- Dropout layer (rate: 0.3)
- Output layer with 10 neurons and softmax activation

## Results

| Model           | Train Accuracy | Test Accuracy |
| --------------- | -------------- | ------------- |
| Base (5 epochs) | ~98%           | ~98%          |
| Overfit (100e)  | ~100%          | ~97.5–98.1%   |
| With Dropout    | ~99.2%         | ~98% ± 0.05   |

- Dropout stabilizes the performance and adds robustness

## Visualizations

Plots of train/test accuracy over 100 epochs are included in the notebook.

## Dataset

The MNIST dataset is included in `keras.datasets`.

## Source

Based on a tutorial from YouTube (Codebasics), expanded by own experiments.

## Next Steps

- Learn about other regularization methods (e.g. L2)
- Build a CNN for improved performance
- Begin handwritten character recognition on custom images

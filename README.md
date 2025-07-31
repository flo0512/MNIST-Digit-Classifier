# MNIST Digit Classifier

This is my first neural network project using TensorFlow and Keras.  
The model learns to recognize handwritten digits from the MNIST dataset.

## What I Learned

- Basics of neural networks (layers, neurons, activation functions, optimizers)
- How to train and evaluate a model with Keras
- Data normalization and model accuracy evaluation
- Overfitting, underfitting and generalization
- Dropout regularization and its effect on model performance
- Comparison of optimizers (SGD vs. Adam) and their impact on learning behavior
- Basics of Tensorboard
- Confusion_matrix & classification metrics
- Basics of Convolutional Neural Networks (CNNs)

## Model Architecture

### Base Model:

- Flatten layer (input shape: 28x28)
- Dense layer with 128 neurons and ReLU activation
- Output layer with 10 neurons and softmax activation

### Overfit Model:

- Same as base, trained with 100 epochs to observe overfitting behavior

### Underfit Model:

- Flatten layer (input shape: 28x28)
- Output layer with 10 neurons and softmax activation

### Dropout Model:

- Dense layer with 128 neurons and ReLU activation
- Dropout layer (rate: 0.3)
- Output layer with 10 neurons and softmax activation

### CNN Model:

- Convolutional Layer(input shape: 28x28) x3
- Pooling Layer x3
- Flatten layer (input shape: 28x28)
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (rate: 0.5)
- Output layer with 10 neurons and softmax activation

### Optimizer Comparison:

- Same architecture, trained with 30 epochs
- Compared performance using SGD and Adam optimizers

| Model          | Train Accuracy | Test Accuracy |
| -------------- | -------------- | ------------- |
| Base (5 e)     | ~98%           | ~98%          |
| Overfit (100e) | ~100%          | ~97.5–98.1%   |
| Underfit (5e)  | ~92.5%         | ~92.5%        |
| With Dropout   | ~99.2%         | ~98% ± 0.05   |
| SGD (30e)      | Lower          | Lower         |
| Adam (30e)     | Higher         | Higher        |
| CNN (20e)      | ~99.5%         | ~98.5%        |

- Dropout stabilizes the performance and adds robustness
- Adam converges faster and achieves higher accuracy than SGD, but shows minor fluctuations in the learning curve

## Visualizations

Plots of train/test accuracy and loss are included in the notebook(s).  
Comparison between optimizers and dropout impact is visualized for better insight.
I used a Tensorboard locally.

## Dataset

The MNIST dataset is included in `keras.datasets`.

## Source

Based on a tutorial from YouTube (Codebasics), expanded by own experiments.

## Next Steps

- Learn about other regularization methods (e.g. L2)
- Begin handwritten character recognition on custom images
- understanding Tensorboard better

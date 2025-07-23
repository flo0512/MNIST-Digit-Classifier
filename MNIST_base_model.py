
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

len(X_train), len(X_test)  # Number of training and test samples
X_train = X_train / 255.0  # Normalize the training images
X_test = X_test / 255.0  # Normalize the test images

model = keras.models.Sequential()  # Create a Sequential model
# Flatten the input images
model.add(keras.layers.Flatten(input_shape=(28, 28)))
# Add a dense layer with ReLU activation
model.add(keras.layers.Dense(128, activation='relu'))
# Add an output layer with softmax activation
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # Compile the model

# Train the model
model.fit(X_train, y_train, epochs=5)  # Train the model for 5
# Evaluate the model
# Evaluate the model on the test set.
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)  # Print the test accuracy

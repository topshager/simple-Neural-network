import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

# Load data from library
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalization of data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Creating model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Corrected loss function name
              metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', test_accuracy)

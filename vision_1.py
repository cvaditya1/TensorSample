import keras.layers
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Dense, Flatten

data = datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

input = Flatten(input_shape=(28, 28))
l0 = Dense(128, activation=tf.nn.relu)
l1 = Dense(10, activation=tf.nn.softmax)
model = Sequential([input, l0, l1])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create model
model.fit(training_images, training_labels, epochs=50)

# Test the model
model.evaluate(test_images, test_labels)

#Exploring
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
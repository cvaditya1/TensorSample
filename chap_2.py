import keras.layers
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, datasets, callbacks
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

data = datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

l0 = Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1))
l1 = MaxPooling2D(2, 2)
l2 = Conv2D(64, (3, 3), activation=tf.nn.relu)
l3 = MaxPooling2D(2, 2)
l4 = Flatten()
l5 = Dense(128, activation=tf.nn.relu)
l6 = Dense(10, activation=tf.nn.softmax)
model = Sequential([l0, l1, l2, l3, l4, l5, l6])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class myCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print('\nReached 95% accuracy so cancelling training')
            self.model.stop_training = True


callback = myCallback()
# Create model
model.fit(training_images, training_labels, epochs=50)

# Test the model
model.evaluate(test_images, test_labels)

# Exploring
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

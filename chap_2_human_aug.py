import keras.layers
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, datasets, callbacks
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
training_dir = 'horse_or_human/training/'
train_generator = train_datagen.flow_from_directory(training_dir, target_size=(300, 300), class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1 / 255)
validation_dir = 'horse_or_human/validation/'
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(300, 300),
                                                              class_mode='binary')

l0 = Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3))
l1 = MaxPooling2D(2, 2)
l2 = Conv2D(32, (3, 3), activation=tf.nn.relu)
l3 = MaxPooling2D(2, 2)
l4 = Conv2D(64, (3, 3), activation=tf.nn.relu)
l5 = MaxPooling2D(2, 2)
l6 = Conv2D(64, (3, 3), activation=tf.nn.relu)
l7 = MaxPooling2D(2, 2)
l8 = Conv2D(64, (3, 3), activation=tf.nn.relu)
l9 = MaxPooling2D(2, 2)
l10 = Flatten()
l11 = Dense(512, activation=tf.nn.relu)
l12 = Dense(1, activation=tf.nn.sigmoid)

model = Sequential([l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, epochs=15, validation_data=validation_generator)

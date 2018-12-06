import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_data_raw, train_labels_raw), (test_data, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_data = train_data_raw / 255
test_data = test_data / 255

validationCount = 10000
validation_labels = train_labels_raw[:validationCount]
validation_data = train_data[:validationCount]
train_labels = train_labels_raw[validationCount:]
train_data = train_data[validationCount:]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(1024, activation=keras.activations.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1024, activation=keras.activations.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, batch_size=100, epochs=20,
          shuffle=True, validation_data=(validation_data, validation_labels))
(loss, acc) = model.evaluate(test_data, test_labels)
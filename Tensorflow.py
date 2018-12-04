import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_data = train_data / 255
test_data = test_data / 255

#refine_labels = train_labels[:5000]
#refine_data = train_data[:5000]

#train_labels = train_labels[5000:]
#train_data = train_data[5000:]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(1024, activation=keras.activations.relu))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=100, epochs=5)
(loss, acc) = model.evaluate(test_data, test_labels)
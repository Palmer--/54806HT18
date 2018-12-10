import tensorflow as tf
import keras
import sipaModels as sipa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preProccess(data):
    return data/255

def preProccessCovModel(data):
    return data.reshape(-1, 28, 28, 1) / 255

fashion_mnist = keras.datasets.fashion_mnist
(train_data_raw, train_labels), (test_data_raw, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_data = preProccessCovModel(train_data_raw)
test_data = preProccessCovModel(test_data_raw)

model = sipa.getConvModel()

history = model.fit(train_data, train_labels, epochs=10,
          shuffle=True, validation_split=0.2)
(loss, acc) = model.evaluate(test_data, test_labels)

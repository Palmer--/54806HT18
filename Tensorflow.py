import tensorflow as tf
import keras
import sipaModels as sipa
import numpy as np
import historyPlot as siplot
import modelSaver as saver
import matplotlib.pyplot as plt

def preProccess(data):
    return data / 255

def preProccessCovModel(data):
    return data.reshape(-1, 28, 28, 1) / 255


fashion_mnist = keras.datasets.fashion_mnist
(train_data_raw, train_labels), (test_data_raw, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

conv_train = preProccessCovModel(train_data_raw)
conv_test = preProccessCovModel(test_data_raw)
train_data = preProccess(train_data_raw)
test_data = preProccess(test_data_raw)

model = sipa.getConvModel()

testAccuracy = []
testLoss = []
trainAccuracy = []
trainLoss = []

for epo in range(20):
    history = model.fit(conv_train, train_labels, epochs=1, shuffle=True, validation_split=0.2)
    trainAccuracy.append(history.history["val_acc"])
    trainLoss.append((history.history["val_loss"]))
    loss, acc = model.evaluate(conv_test, test_labels)
    testAccuracy.append(acc)
    testLoss.append(loss)

testAccuracy = np.asanyarray(testAccuracy)
testLoss = np.asanyarray(testLoss)
trainAccuracy = np.asanyarray(trainAccuracy)
trainLoss = np.asanyarray(trainLoss)

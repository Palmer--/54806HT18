import sipaModels as sipa
import keras
import  matplotlib.pyplot as plt
import historyPlot as siplot
import numpy as np

conv_train = sipa.getConvTrainData()
conv_test = sipa.getConvTestData()
train_labels = sipa.getTrainLabels()
test_labels = sipa.getTestLabels()

def evaluateModel(model, epocs, name):
    testAccuracy = []
    testLoss = []
    trainAccuracy = []
    trainLoss = []

    for epo in range(epocs):
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
    siplot.savePlot([trainAccuracy, testAccuracy], name + " accuracy", "accuracy", ["Train", "Test"])
    siplot.savePlot([trainLoss, testLoss], name + " loss", "loss", ["Train", "Test"])

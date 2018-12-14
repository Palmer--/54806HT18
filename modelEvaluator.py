import sipaModels as sipa
import keras
import  matplotlib.pyplot as plt
import historyPlot as siplot
import numpy as np
import datetime as dt

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
        trainAccVal = history.history["val_acc"]
        trainAccuracy.append(trainAccVal)
        trainLossVal = history.history["val_loss"]
        trainLoss.append(trainLossVal)
        loss, acc = model.evaluate(conv_test, test_labels)
        testAccuracy.append(acc)
        testLoss.append(loss)

    testAccuracy = np.asanyarray(testAccuracy)
    testLoss = np.asanyarray(testLoss)
    trainAccuracy = np.asanyarray(trainAccuracy)
    trainLoss = np.asanyarray(trainLoss)
    siplot.savePlot([trainAccuracy, testAccuracy], name + " accuracy", "accuracy", ["Train data", "Test data"])
    siplot.savePlot([trainLoss, testLoss], name + " loss", "loss", ["Train data", "Test data"])
    writeResultsToFile(name, testAccuracy, testLoss, trainAccuracy, trainLoss)


def writeResultsToFile(modelName, testAccuracy, testLoss, trainAccuracy, trainLoss):
    maxTestAccValue = max(testAccuracy)
    maxTestAccIndex = np.where(testAccuracy == maxTestAccValue)[0][0]
    minTestLossValue = min(testLoss)
    minTestLossIndex = np.where(testLoss == minTestLossValue)[0][0]

    maxTrainAccValue = max(trainAccuracy)
    maxTrainAccIndex = np.where(trainAccuracy == maxTrainAccValue)[0][0]
    minTrainLossValue = min(trainLoss)
    minTrainLossIndex = np.where(trainLoss == minTrainLossValue)[0][0]
    with open("Results " + modelName + ".txt", "a") as file:
        file.write("##%s###\n" % modelName)
        file.write("Created at %s\n" % dt.datetime.now())
        file.write("Highest Test Accuracy: %s at epoch: %s\n" % (maxTestAccValue, maxTestAccIndex))
        file.write("Lowest Test Loss: %s at epoch: %s\n" % (minTestLossValue, minTestLossIndex))
        file.write("Highest Train Accuracy: %s at epoch: %s\n" % (maxTrainAccValue, maxTrainAccIndex))
        file.write("Lowest Train Loss: %s at epoch: %s\n" % (minTrainLossValue, minTrainLossIndex))



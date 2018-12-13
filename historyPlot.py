import keras
import matplotlib.pyplot as plt
import numpy as np

def plotAccuracyHistory(history: [keras.models], legend: [str]):
    plt.figure()
    for x in history:
        plt.plot(x.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend, loc='best')
    plt.show()


def plotLossHistory(history):
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.grid(color="gray", linestyle='-', linewidth="0.25")
    plt.legend(['Loss'], loc='best')
    plt.show()


def savePlot(plots, title, ylabel, legend):

    plt.figure()
    for index, plot in enumerate(plots):
        plt.plot(plot)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(legend, loc='best')
    plt.grid(color="gray", linestyle='-', linewidth="0.25")
    plt.savefig(title)
    plt.close()

def getMax(line):
    ymax = max(line)
    xpos = line.argmax()
    return xpos, ymax

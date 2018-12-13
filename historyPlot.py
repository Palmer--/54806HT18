import keras
import matplotlib.pyplot as plt

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
    plt.legend(['Loss'], loc='best')
    plt.show()


def savePlot(plots, title, ylabel, legend):
    plt.figure()
    for line in plots:
        plt.plot(line)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(legend, loc='best')
    plt.savefig(title)
    plt.close()

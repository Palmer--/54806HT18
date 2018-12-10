import keras
import matplotlib as plt

def plotAccuracyHistory(history: keras.models):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Accuracy'], loc='top right')
    plt.show()

def plotLossHistory(history):
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Loss'], loc='top right')
    plt.show()


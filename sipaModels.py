import keras
import tensorflow as tf

fashion_mnist = keras.datasets.fashion_mnist
(train_data_raw, train_labels), (test_data_raw, test_labels) = fashion_mnist.load_data()

def defaultCompile(model):
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


def preProccess(data):
    return data / 255


def preProccessCovData(data):
    return data.reshape(-1, 28, 28, 1) / 255


def getConvTrainData():
    return preProccessCovData(train_data_raw)


def getConvTestData():
    return preProccessCovData(test_data_raw)


def getTrainLabels():
    return train_labels


def getTestLabels():
    return test_labels


def getTutorialModel():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
    defaultCompile(model)
    return model


def getTutorialModelMultiLayer():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
    defaultCompile(model)
    return model


def getBaseModel():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=([28, 28, 1])),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (5, 5), padding="same"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')])
    defaultCompile(model)
    return model


def getConvModelSmallKernel():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (2, 2), padding="same", input_shape=([28, 28, 1])),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (12, 12), padding="same"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')])
    defaultCompile(model)
    return model


def getConvModelBigKernel():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (14, 14), padding="same", input_shape=([28, 28, 1])),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (2, 2), padding="same"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')])
    defaultCompile(model)
    return model

def getConvModelValidPadding():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (14, 14), padding="same", input_shape=([28, 28, 1])),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (2, 2), padding="same"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')])
    defaultCompile(model)
    return model

def getConvModelBigStrides():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (5, 5), strides=2, padding="same", input_shape=([28, 28, 1])),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (5, 5), padding="same"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')])
    defaultCompile(model)
    return model


def getConvModelMultiLayer():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=([28, 28, 1])),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (5, 5), padding="same"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')])
    defaultCompile(model)
    return model

def getConvModelNoDropout():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=([28, 28, 1])),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (5, 5), padding="same"),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(10, activation='softmax')])
    defaultCompile(model)
    return model
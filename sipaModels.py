import keras
import tensorflow as tf

def defaultCompile(model):
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

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

def getConvModel():
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


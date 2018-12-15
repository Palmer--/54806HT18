import keras
import keras.preprocessing.image as gen
import numpy as np

def generateMoarData(data):
    retval = np.zeros(data.shape)
    datagen = gen.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    for index, d in enumerate(data):
        retval[index] = datagen.random_transform(d)
    return retval

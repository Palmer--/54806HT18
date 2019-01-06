import keras
import keras.layers
from keras.layers import *
import numpy as np
import NChain as nc
import random as rand


def get_model():
    input_tensor = keras.layers.Input((3,))
    hidden_layer1 = Dense(units=128, activation='relu')(input_tensor)
    output_tensor = Dense(units=3, activation='linear')(hidden_layer1)
    model = keras.Model(input_tensor, output_tensor)
    model.compile(loss='mse', optimizer='adam')
    return model


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_random_action(action_space: int):
    space = np.zeros(action_space)
    space[rand.randint(0, len(space)-1)] = 1
    return space

def create_training_data():
    memory = []
    random = rand.Random()
    for round in range(100000):
        previous_observations = []
        env = nc.Nchain()
        done = False
        total_score = 0
        while not done:
            state = env.get_state()
            action = get_random_action(env.get_action_space())
            reward, done = env.make_move(action)
            total_score += reward
            previous_observations.append((state, action))
        memory.append((previous_observations, total_score))
    memory.sort(key=lambda mem: mem[1])
    return memory[-10000:]

model = get_model()

data = create_training_data()[0]
train_data = data[0]
train_labels = data[1]
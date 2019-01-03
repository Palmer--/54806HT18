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

def get_random_action(action_space):
    space = np.random.rand(action_space)
    return np.eye(action_space)[space]

def create_training_data():
    memory = []
    random = rand.Random()
    for round in range(100000):
        previous_observations = []
        env = nc.Nchain()
        done = False
        totalScore = 0
        while not done:
            state = env.get_state()
            action = random.randrange(0, 3)
            reward, done = env.make_move(action)
            totalScore += reward
            nextState = env.get_state()
            previous_observations.append(state, action)
        memory.append(previous_observations, totalScore)
    return memory




model = get_model()

data = np.array([[1, 2, 3]])
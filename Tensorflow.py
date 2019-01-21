import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Flatten
from keras.optimizers import Adam

import keras
from dqn import DQNAgent
from policy import *
from memory import SequentialMemory
import snakeGame as sg

keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


# Get the environment and extract the number of actions.
env = sg.SnakeGame()
np.random.seed(123)
env.seed(123)
visualize_training = False
nb_actions = env.action_space.n

# build model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(432, activation='relu'))
model.add(Dense(216, activation='relu'))
model.add(Dense(144, activation='linear'))
model.add(Dense(nb_actions))
print(model.summary())

memory = SequentialMemory(limit=500000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-2), metrics=['mae'])

dqn.load_weights('dqn_Snake_weights_2_weights.h5f')
#dqn.fit(env, nb_steps=500000, visualize=visualize_training, verbose=2, callbacks=[tbCallBack])
# After training is done save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format("Snake_weights_2"), overwrite=True)

# evaluate our model for 5 episodes.
dqn.test(env, nb_episodes=500, visualize=True)


import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Flatten
from keras.optimizers import Adam

from dqn import DQNAgent
from policy import *
from memory import SequentialMemory
import snakeGame as sg


# Get the environment and extract the number of actions.
env = sg.SnakeGame()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(432, activation='relu'))
model.add(Dense(432, activation='relu'))
model.add(Dense(432, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.3)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
#dqn.save_weights('dqn_{}_weights.h5f'.format("Snake_weights"), overwrite=True)
dqn.load_weights('dqn_Snake_weights_weights_1.h5f')

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=500, visualize=True)

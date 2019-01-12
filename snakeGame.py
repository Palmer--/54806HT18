#SNAKES GAME
#SOURCE: https://gist.github.com/sanchitgangwar/2158089
import numpy as np
import os
import random
from random import randint
import gym.spaces

Movement_Map = ["Up", "Right", "Down", "Left"]
Snake_Head_Val = 1
Snake_Body_Val = 2
Food_Val = 3
Reward_EatFood = 1.0
Reward_Move = -0.01
Reward_Lose = -10.0

class SnakeGame:
    _init_snake = [(4, 4), (4, 5), (4, 6)]
    world_dimensions = (10, 10)
    snake = None # Initial snake co-ordinates
    food = None
    score = 0.0
    _seed = None

    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

    @property
    def snake_head(self):
        return self.snake[0]

    @property
    def snake_body(self):
        return self.snake[1:]

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        random.seed(self._seed)

    def reset(self):
        self.snake = self._init_snake.copy()
        self.place_food()
        self.score = 0.0
        return self.get_state()

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 1, self.world_dimensions, 'float32')

    def seed(self, value):
        self._seed = value

    def get_state(self):
        state = np.zeros(self.world_dimensions)
        state[self.snake_head] = Snake_Head_Val
        for part in self.snake[1:]:
            state[part] = Snake_Body_Val
        for food_bit in self.food:
            state[food_bit] = Food_Val
        return state

    def step(self, action: int):
        if len(action) is not len(Movement_Map):
            raise ValueError("parameter action must have a length of {} but got a length of {}", len(action), len(Movement_Map))
        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].
        action_reward = Reward_Move
        done = False
        self.snake.insert(0, (self.snake[0][0] + (action[2] and 1) + (action[0] and -1),
                              self.snake[0][1] + (action[3] and -1) + (action[1] and 1)))

        # Exit if snake crosses the boundaries
        if self.snake[0][0] < 0 or self.snake[0][0] >= self.world_dimensions[0] \
                or self.snake[0][1] < 0 or self.snake[0][1] >= self.world_dimensions[1]:
            action_reward = Reward_Lose
            done = True

        # If snake runs over itself
        if self.snake[0] in self.snake[1:]:
            action_reward = Reward_Lose
            done = True

        # When snake eats the food
        if self.snake[0] in self.food:
            action_reward = Reward_EatFood
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()  # [1] If it does not eat the food, length decreases

        return self.get_state(), action_reward, done, None

    def place_food(self):
        self.food = None
        while True:
            maybe_food = (randint(1, self.world_dimensions[0] - 2),
                          randint(1, self.world_dimensions[1] - 2))  # Calculating next food's coordinates
            if maybe_food not in self.snake:
                self.food = [maybe_food]
                break

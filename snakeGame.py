#SNAKES GAME
#SOURCE: https://gist.github.com/sanchitgangwar/2158089
import numpy as np
import os
import random
from random import randint
import gym.spaces

Movement_Map = ["Up", "Right", "Down", "Left"]
# Values used in the simple model
Snake_Head_Val = 1
Snake_Body_Val = 2
Food_Val = 3

Reward_EatFood = 1.0  # Given when the snake successfully eats food
Reward_Move = 0.0  # Default movement reward
Reward_Invalid_Move = -1.0  # Given when an invalid move is made
Reward_Lose = -1.0  # Given when the snake runs into itself or a wall
Reward_Approach = 0.1  # Given when the snake moves closer to food


class SnakeGame:
    _init_snake = [(4, 4), (4, 5), (4, 6)]
    world_dimensions = (11, 11) #This will allow for 10x10 gamespace since the edges are considered game over
    snake = _init_snake.copy()
    food = []
    score = 0.0
    _seed = None
    previous_action = 0
    renderer = None
    step_limit = 200
    steps = 0

    def __init__(self):
        pass

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
        self.previous_action = 0
        self.steps = 0

        return self.get_state()

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 1, np.shape(self.get_state()), 'float32')

    def seed(self, value):
        self._seed = value

    @property
    def valid_actions(self):
        valid_actions = [x for x in range(len(Movement_Map))]
        valid_actions.remove((self.previous_action+2) % 4)
        # Do not allow the snake leave bounds
        if self.snake_head[0] == 0:
            valid_actions.remove(3)
        if self.snake_head[0] == self.world_dimensions[0]-1:
            valid_actions.remove(1)
        if self.snake_head[1] == 0:
            valid_actions.remove(0)
        if self.snake_head[1] == self.world_dimensions[1]-1:
            valid_actions.remove(2)
        return valid_actions

    def get_state(self):
        return self.get_snake_head_state(), self.get_snake_body_state(), self.get_food_state()

    def get_simple_state(self):
        state = np.zeros(self.world_dimensions)
        state[self.snake_head] = Snake_Head_Val
        for part in self.snake_body:
            state[part] = Snake_Body_Val
        for bit in self.food:
            state[bit] = Food_Val
        return state

    def get_snake_head_state(self):
        state = np.zeros(self.world_dimensions)
        state[self.snake_head] = 1
        return state

    def get_snake_body_state(self):
        state = np.zeros(self.world_dimensions)
        for part in self.snake_body:
            state[part] = 1
        return state

    def get_food_state(self):
        state = np.zeros(self.world_dimensions)
        for bit in self.food:
            state[bit] = 1
        return state

    def _get_next_snake_head(self, action: int, current_head):
        if action == 0:
            return current_head[0], current_head[1]-1
        if action == 1:
            return current_head[0]+1, current_head[1]
        if action == 2:
            return current_head[0], current_head[1]+1
        if action == 3:
            return current_head[0]-1, current_head[1]

        raise ValueError("Unkown action: {}", action)

    def render(self, mode='human', close=False):
        import SnakeGameRenderer as drawer
        if self.renderer is None:
            self.renderer = drawer.SnakeGameDrawer(self)
        self.renderer.draw_state(self)

    def step(self, action: int):
        self.steps += 1
        done = False
        if self.step_limit <= self.steps:
            done = True

        # Snake head can not go back to the same location is just came from
        if action not in self.valid_actions:
            return self.get_state(), Reward_Invalid_Move, done, dict()
        next_head = self._get_next_snake_head(action, self.snake_head)

        self.snake.insert(0, next_head)
        self.previous_action = action

        # Exit if snake reaches edge
        if next_head[0] < 1 or next_head[0]+1 >= self.world_dimensions[0] \
                or next_head[1] < 1 or next_head[1]+1 >= self.world_dimensions[1]:
            return self.get_state(), Reward_Lose, True, dict()

        # If snake runs over itself
        if self.snake_head in self.snake[1:]:
            return self.get_state(), Reward_Lose, True, dict()

        # When snake eats the food
        if next_head in self.food:
            self.place_food()
            return self.get_state(), Reward_EatFood, done, dict()
        else:
            self.snake.pop()  # If it does not eat the food, length decreases
            prev_head = self.snake_body[0]
            prev_dist = self.dist(self.food[0], prev_head)
            new_dist = self.dist(self.food[0], next_head)
            if new_dist < prev_dist:
                return self.get_state(), Reward_Move, done, dict()
            else:
                return self.get_state(), Reward_Approach, done, dict()

    @staticmethod
    def dist(x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    def place_food(self):
        self.food = None
        while True:
            maybe_food = (randint(1, self.world_dimensions[0] - 2),
                          randint(1, self.world_dimensions[1] - 2))  # Calculating next food's coordinates
            if maybe_food not in self.snake:
                self.food = [maybe_food]
                break

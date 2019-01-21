#SNAKES GAME
#SOURCE: https://gist.github.com/sanchitgangwar/2158089
import numpy as np
import os
import random
from random import randint
import gym.spaces

Movement_Map = ["Up", "Right", "Down", "Left"]
Input_Map = ["Left", "Straight", "Right"]
# Values used in the simple model
Snake_Body_Val = 1 / 4 * -2
Snake_Head_Val = 1/4 * 1
Wall_Val = 1/4 * -4
Food_Val = 1/4 * 4

Reward_EatFood = 1.0  # Given when the snake successfully eats food
Reward_Invalid_Move = 0  # Given when an invalid move is made
Reward_Lose = -1.0  # Given when the snake runs into itself or a wall
Reward_Correct_Direction = 0.1  # Given when the snake moves closer to food
Reward_Wrong_Direction = 0.1 #Given when the snake moves away from food
world_size = 7  # This will allow for (n-2)^2 gamespace since the edges are considered game over
step_limit = 100  # max number of steps per epoc


class SnakeGame:

    def __init__(self):
        self.food = []
        self.steps = 0
        self.previous_action = 1
        self.world_size = world_size
        self.snake = []
        self.renderer = None
        self._seed = None
        self.score = 0.0
        self.rotation = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(3)

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
        snake_start = self.get_random_cordinates()
        self.snake = [snake_start] * 3
        self.place_food()
        self.score = 0.0
        self.previous_action = 2
        self.steps = 0
        self.rotation = 0
        return self.get_state()

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 2, (3, self.world_size, self.world_size), 'float32')

    def seed(self, value):
        self._seed = value

    @property
    def valid_actions(self):
        valid_actions = [x for x in range(len(Movement_Map))]
        if self.previous_action is not None:
            valid_actions.remove((self.previous_action+2) % 4)
        # Do not allow the snake leave bounds
        if self.snake_head[0] == 0:
            valid_actions.remove(3)
        if self.snake_head[0] == self.world_size-1:
            valid_actions.remove(1)
        if self.snake_head[1] == 0:
            valid_actions.remove(0)
        if self.snake_head[1] == self.world_size-1:
            valid_actions.remove(2)
        return valid_actions

    def get_state(self):
        return self.get_complex_state()

    def get_complex_state(self):
        state = np.zeros((3, self.world_size, self.world_size))
        state[0] = np.rot90(self.get_snake_head_state(), self.rotation)
        state[1] = np.rot90(self.get_snake_body_state(), self.rotation)
        state[2] = np.rot90(self.get_food_state(), self.rotation)
        return state

    def get_simple_state(self):
        state = np.zeros((self.world_size, self.world_size))
        for part in self.snake_body:
            state[part] = Snake_Body_Val
        for bit in self.food:
            state[bit] = Food_Val
        state[self.snake_head] = Snake_Head_Val
        return np.rot90(state, self.rotation)

    def get_snake_head_state(self):
        state = np.zeros((self.world_size, self.world_size))
        state[self.snake_head] = 1
        return state

    def get_snake_body_state(self):
        state = np.zeros((self.world_size, self.world_size))
        for part in self.snake_body:
            state[part] = 1
        return state

    def get_food_state(self):
        state = np.zeros((self.world_size, self.world_size))
        for bit in self.food:
            state[bit] = 1
        return state

    @staticmethod
    def _get_next_snake_head(action: int, current_head):
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
        import SnakeGameRenderer as sgr
        if self.renderer is None:
            self.renderer = sgr.SnakeGameDrawer(self)
        self.renderer.draw_state(self)

    def step(self, raw_action: int):
        self.steps += 1
        done = False
        if step_limit <= self.steps:
            done = True
        action = (raw_action + 1 + self.rotation) % 4

        # Snake head can not go back to the same location is just came from
        if action not in self.valid_actions:
            raise ValueError("Snake can not go that way")
        next_head = self._get_next_snake_head(action, self.snake_head)

        # We have made a valid move and can update the snake position
        prev_head = self.snake_head
        self.snake.insert(0, next_head)
        self.previous_action = action
        self.rotation += raw_action - 1

        # Exit if snake reaches edge
        if next_head[0] < 1 or next_head[0]+1 >= self.world_size \
                or next_head[1] < 1 or next_head[1]+1 >= self.world_size:
            return self.get_state(), Reward_Lose, True, dict()

        # If snake runs over itself
        if next_head in self.snake_body:
            return self.get_state(), Reward_Lose, True, dict()

        # When snake eats the food
        if next_head in self.food:
            self.place_food()
            return self.get_state(), Reward_EatFood, done, dict()
        else:
            self.snake.pop()  # If it does not eat the food, length decreases
            prev_dist = self.dist(self.food[0], prev_head)
            new_dist = self.dist(self.food[0], next_head)
            if new_dist < prev_dist:
                return self.get_state(), Reward_Wrong_Direction, done, dict()
            else:
                return self.get_state(), Reward_Correct_Direction, done, dict()

    @staticmethod
    def dist(x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    def get_random_cordinates(self):
        return randint(1, self.world_size - 2), randint(1, self.world_size - 2)

    def place_food(self):
        self.food = None
        while True:
            maybe_food = self.get_random_cordinates()  # Calculating next food's coordinates
            if maybe_food not in self.snake:
                self.food = [maybe_food]
                break

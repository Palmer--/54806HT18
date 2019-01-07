#SNAKES GAME
#SOURCE: https://gist.github.com/sanchitgangwar/2158089
import pygame as pg
import numpy as np
import os
from random import randint

Movement_Map = ["Up", "Right", "Down", "Left"]
Snake_Head_Val = 1
Snake_Body_Val = 2
Food_Val = 3
Reward_EatFood = 1.0
Reward_Move = -0.01
Reward_Lose = -10.0

class SnakeGame:
    world_dimensions = (10, 10)
    snake = [(4, 10), (4, 9), (4, 8)]  # Initial snake co-ordinates
    food = (10, 20)  # First food co-ordinates
    score = 0.0
    previous_input = np.array([0, 0, 0, 0])

    #Draw stuff
    draw_resolution = 20
    screen_dim = (world_dimensions[0]*draw_resolution,
                  world_dimensions[1]*draw_resolution)
    snake_color = pg.Color("green")
    food_color = pg.Color("red")
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    screen = pg.display.set_mode(screen_dim)

    def snake_head(self):
        return self.snake[0]

    def get_state(self):
        state = np.zeros(self.world_dimensions)
        state[self.snake_head()] = Snake_Head_Val
        for snake_body in self.snake[1:]:
            state[snake_body] = Snake_Body_Val
        for food_bit in self.food:
            state[food_bit] = Food_Val
        return state

    def draw_state(self):
        surface = pg.Surface(self.screen_dim)
        for x in self.snake:
            surface = pg.Surface(self.draw_resolution)
            surface.fill(self.snake_color)
            pg.draw.rect(surface, pg.Color.)
        self.screen.

    def step(self, action: np.ndarray):
        if len(action) is not len(Movement_Map):
            raise ValueError("parameter action must have a length of {} but got a length of {}", len(action), len(Movement_Map))
        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].
        action_reward = -0.01
        done = False
        self.snake.insert(0, (self.snake[0][0] + (action[2] and 1) + (action[0] and -1),
                              self.snake[0][1] + (action[3] and -1) + (action[1] and 1)))

        # Exit if snake crosses the boundaries
        if self.snake[0][0] == 0 or self.snake[0][0] == 19 or self.snake[0][1] == 0 or self.snake[0][1] == 59:
            action_reward = -1
            done = True

        # If snake runs over itself
        if self.snake[0] in self.snake[1:]:
            action_reward = -1
            done = True

        # When snake eats the food
        if self.snake[0] in self.food:
            action_reward = 1
            self.food = []
            self.score += 1
            while self.food is None:
                self.food = (randint(1, self.world_dimensions[0]-2), randint(1, self.world_dimensions[1]-2))  # Calculating next food's coordinates
                if food in self.snake:
                    food = None
        else:
            self.snake.pop()  # [1] If it does not eat the food, length decreases

        return self.get_state(), action_reward, done, ""
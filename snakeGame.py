#SNAKES GAME
#SOURCE: https://gist.github.com/sanchitgangwar/2158089
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting
import pygame
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

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.init()
    screen = pygame.display.set_mode((200, 200))

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

    def step(self, action: np.ndarray):
        if len(action) is not len(Movement_Map):
            raise ValueError("parameter action must have a length of {} but got a length of {}", len(action), len(Movement_Map))

        
        return None

    while key != 27:  # While Esc key is not pressed
        #win.border(0)
        #win.addstr(0, 2, 'Score : ' + str(score) + ' ')  # Printing 'Score' and
        #win.addstr(0, 27, ' SNAKE ')  # 'SNAKE' strings
        #win.timeout(150 - (len(snake) / 5 + len(snake) / 10) % 120)  # Increases the speed of Snake as its length increases

        prevKey = key  # Previous key pressed
        key = key if event == -1 else event

        if key == ord(' '):  # If SPACE BAR is pressed, wait for another
            key = -1  # one (Pause/Resume)
            while key != ord(' '):
                key = win.getch()
            key = prevKey
            continue

        #if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:  # If an invalid key is pressed
        #    key = prevKey

        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        # This is taken care of later at [1].
        snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1),
                         snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])

        # Exit if snake crosses the boundaries
        if snake[0][0] == 0 or snake[0][0] == 19 or snake[0][1] == 0 or snake[0][1] == 59:
            break

        # If snake runs over itself
        if snake[0] in snake[1:]:
            break

        if snake[0] == food:  # When snake eats the food
            food = []
            score += 1
            while food is None:
                food = (randint(1, world_dimensions[0]-2), randint(1, world_dimensions[1]-2))  # Calculating next food's coordinates
                if food in snake: food = None
        else:
            last = snake.pop()  # [1] If it does not eat the food, length decreases
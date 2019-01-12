# Import a library of functions called 'pygame'
import pygame as pg
import numpy as np
import snakeGame as sg
import os

class SnakeGameDrawer():
# Initialize the game engine
    pg.init()
    screen = None

    def __init__(self, game: sg.SnakeGame):
        self.screen = pg.display.set_mode()
        pg.display.set_caption("ML Snake")
        # Draw stuff

        screen_dim = (game.world_dimensions[0] * self.DRAW_RESOLUTION,
                      game.world_dimensions[1] * self.DRAW_RESOLUTION)
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pg.init()
        self.screen = pg.display.set_mode(screen_dim)

    # Define the colors we will use in RGB format
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    DRAW_RESOLUTION = 20

    # Set the height and width of the screen
    size = [400, 300]
    @staticmethod
    def to_draw_rect(location, resolution):
        return [location[0]*resolution, location[1]*resolution, resolution, resolution]

    def play(self):
        # Loop until the user clicks the close button.
        done = False
        clock = pg.time.Clock()

        while not done:

            # This limits the while loop # of updates per sec.
            clock.tick(3)
            self.screen.fill(self.WHITE)


    def draw_state(self, game : sg.SnakeGame):
        self.screen.fill(self.WHITE)
        head_rect = self.to_draw_rect(game.snake_head(), self.DRAW_RESOLUTION)
        pg.draw.rect(self.screen, self.BLACK, head_rect)
        for bit in game.food:
            rect = self.to_draw_rect(bit, self.DRAW_RESOLUTION)
            pg.draw.rect(self.screen, self.RED, rect)

        for part in game.snake:
            part_rect = self.to_draw_rect(part, self.DRAW_RESOLUTION)
            pg.draw.rect(self.screen, self.RED, part_rect)
            pg.display.flip()

        # Be IDLE friendly
        pg.quit()
# Import a library of functions called 'pygame'
import pygame as pg
import numpy as np
import snakeGame as sg
import os

class SnakeGameDrawer():
# Initialize the game engine
    pg.init()
    screen = None

    def __init__(self, game: sg.SnakeGame, history: np.ndarray):
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

    def play(self):
        # Loop until the user clicks the close button.
        done = False
        clock = pg.time.Clock()

        while not done:

            # This limits the while loop # of updates per sec.
            clock.tick(3)
            self.screen.fill(self.WHITE)

            # Draw on the screen a GREEN line from (0,0) to (50.75)
            # 5 pixels wide.
            pg.draw.line(screen, GREEN, [0, 0], [50, 30], 5)

            # Draw on the screen a GREEN line from (0,0) to (50.75)
            # 5 pixels wide.
            pg.draw.lines(screen, BLACK, False, [[0, 80], [50, 90], [200, 80], [220, 30]], 5)

            # Draw on the screen a GREEN line from (0,0) to (50.75)
            # 5 pixels wide.
            pg.draw.aaline(screen, GREEN, [0, 50], [50, 80], True)

            # Draw a rectangle outline
            pg.draw.rect(screen, BLACK, [75, 10, 50, 20], 2)

            # Draw a solid rectangle
            pg.draw.rect(screen, BLACK, [150, 10, 50, 20])

            # Draw an ellipse outline, using a rectangle as the outside boundaries
            pg.draw.ellipse(screen, RED, [225, 10, 50, 20], 2)

            # Draw an solid ellipse, using a rectangle as the outside boundaries
            pg.draw.ellipse(screen, RED, [300, 10, 50, 20])

            # This draws a triangle using the polygon command
            pg.draw.polygon(screen, BLACK, [[100, 100], [0, 200], [200, 200]], 5)

            # Go ahead and update the screen with what we've drawn.
            # This MUST happen after all the other drawing commands.
            pg.display.flip()

        # Be IDLE friendly
        pg.quit()
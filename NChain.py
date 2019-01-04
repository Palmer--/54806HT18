import numpy as np
import random

class Nchain:
    length = 5
    currentPos = 0
    round_limit = 8
    current_round = 0

    @staticmethod
    def get_action_space():
        return 3

    def __init__(self):
        self.length = random.randrange(1, 10)
        self.round_limit = random.randrange(1, 10)

    def init_field(self, chain_length):
        self.length = chain_length

    def reset(self):
        self.currentPos = 0
        self.current_round = 0

    def make_move(self, action: np.ndarray):
        self.current_round += 1
        done = self.current_round == self.round_limit
        if action[0] == 1:
            return self.move_left(), done
        if action[1] == 1:
            return self.wait_here(), done
        if action[2] == 1:
            return self.move_right(), done
        raise ValueError('Unknown action.')


    def move_left(self):
        self.currentPos -= 1
        if self.currentPos < 0:
            self.currentPos = 0
            return 1
        return 0

    def wait_here(self):
        return 0

    def move_right(self):
        self.currentPos += 1
        if self.currentPos > self.length:
            self.currentPos = self.length
            return 10
        return 0

    def get_state(self):
        return np.array([self.currentPos, self.length, self.round_limit])


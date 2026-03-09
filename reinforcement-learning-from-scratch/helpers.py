"""Shared helpers for reinforcement learning scripts."""
import numpy as np


class Gridworld:
    """A 5x5 grid: agent navigates from start to goal, avoiding walls."""
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.walls = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)}
        self.actions = {0: (-1, 0), 1: (1, 0),   # up, down
                        2: (0, -1), 3: (0, 1)}    # left, right
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self.actions[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        # Stay in place if hitting a wall or going out of bounds
        if (0 <= nr < self.size and 0 <= nc < self.size
                and (nr, nc) not in self.walls):
            self.state = (nr, nc)
        reward = 1.0 if self.state == self.goal else -0.01
        done = self.state == self.goal
        return self.state, reward, done

    def render(self):
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.state:    row += " A "
                elif (r, c) == self.goal:   row += " G "
                elif (r, c) in self.walls:  row += " # "
                else:                       row += " . "
            print(row)

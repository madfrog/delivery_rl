# coding=utf-8

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from enum import Enum
import json

REWARD_NORMAL = 1
REWARD_COLLISION = -10
REWARD_EDGE_OR_STOP = 0
REWARD_TIMEOUT_EXTRA = -1
REWARD_NORMAL = 2
REWARD_DELIVERY_SUCCESS = 20


class DeliveryEnv(Env):
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.observation_space = Discrete(self.rows * self.cols)
        # Move east: 0, west: 1, south: 2, north: 3, stop: 4
        self.action_space = Discrete(5)
        # Init the position of obstacles
        self.obstacles = random.sample(range(2, 15), 2)
        # Start position
        while True:
            self.state = random.sample(range(0, self.cols), 1)[0]
            self.start = self.state
            if self.start not in self.obstacles:
                break
        # Set destination
        while True:
            self.dest = random.sample(range(self.cols*2, self.cols*3), 1)[0]
            if self.dest not in self.obstacles:
                break
        # Set package time limit
        self.package_time = random.randint(int((self.dest-self.start)/2), self.cols*self.rows)
        self.time_counter = 0

    def reconstruct_env(self):
        with open('./freeze_env.json', 'r') as f:
            info = json.load(f)
            self.rows = info['rows']
            self.cols = info['cols']
            self.observation_space = Discrete(self.rows * self.cols)
            self.action_space = Discrete(info['action_space'])
            self.obstacles = info['obstacles']
            self.start = info['start']
            self.dest = info['dest']
            self.package_time = info['timeout']

    def freeze(self):
        info = {
            "rows": self.rows,
            "cols": self.cols,
            "action_space": 5,
            "obstacles": self.obstacles,
            "start": self.start,
            "dest": self.dest,
            "timeout": self.package_time
        }
        with open('./freeze_env.json', 'w') as f:
            json.dump(info, f)
        print('save env to file success!')


    def __check_edge(self, action):
        # 1. Check if reach the edge, if so, will not move
        # 2. Check if collsion with obstacles, if so, done
        # 3. Check if reach the destination, if so, done
        can_move = True
        if action == 0:
            if (self.state%self.cols + 1) >= self.cols:
                can_move = False
        elif action == 1:
            if self.state%self.cols < 1:
                can_move = False
        elif action == 2:
            if (self.state + self.cols) >= (self.cols * self.rows):
                can_move = False
        elif action == 3:
            if (self.state - self.cols) < 0:
                can_move = False
        return can_move

    def __check_timeout(self):
        return True if self.time_counter > self.package_time else False
    
    def __check_delivery_success(self):
        return True if self.state == self.dest else False

    def __check_collision(self):
        return True if self.state in self.obstacles else False

    def __move(self, action):
        if action == 0:
            # move east
            self.state += 1
        elif action == 1:
            # move west
            self.state -= 1
        elif action == 2:
            # move south
            self.state += self.cols
        elif action == 3:
            # move north
            self.state -= self.cols
        else:
            # stop
            pass


    def step(self, action):
        reward = 0
        done = False
        can_move = self.__check_edge(action)
        if can_move:
            self.__move(action)
            reward = REWARD_NORMAL
        else:
            # can't move
            reward = REWARD_EDGE_OR_STOP

        if self.__check_timeout():
            reward -= REWARD_TIMEOUT_EXTRA
        if self.__check_collision():
            done = True
            reward = REWARD_COLLISION
        if self.__check_delivery_success():
            done = True
            reward = REWARD_DELIVERY_SUCCESS

        info = {}
        return self.state, reward, done, info

    def render(self):
        print("="*20)
        print(f"start position: {self.state}")
        print(f"obstacles: {self.obstacles}")
        print(f"dest: {self.dest}")
        print(f"package time limit: {self.package_time}")
        print("="*20)
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                position = r * self.cols + c
                if position == self.start:
                    row += " S"
                elif position in self.obstacles:
                    row += " X"
                elif position == self.dest:
                    row += " D"
                elif position == self.state:
                    row += " C"
                else:
                    row += " ."
            print(row)
                    
    def reset(self):
        self.observation_space = Discrete(self.rows * self.cols)
        self.action_space = Discrete(5)
        while True:
            self.state = random.sample(range(0, self.cols), 1)[0]
            self.start = self.state
            if self.state not in self.obstacles:
                break
        # self.obstacles = random.sample(range(2, 15), 4)
        while True:
            self.dest = random.sample(range(self.cols*2, self.cols*3), 1)[0]
            if self.dest not in self.obstacles:
                break
        # Set package time limit
        self.package_time = random.randint(int((self.dest-self.start)/2), self.cols*self.rows)
        self.time_counter = 0
        return self.state


if __name__=="__main__":
    env = DeliveryEnv()
    env.render()

    print(f'after freeze...')
    env.freeze()
    env.reconstruct_env()
    env.render()
"""
Reinforcement learning maze example.

"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10 # grid width
OBSTACLE_MAT = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry("+{xPos}+{yPos}".format(xPos=0, yPos=0))  # 控制位置
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])
        # 障碍物
        # obstacle
        self.obstacle = []
        for row in range(OBSTACLE_MAT.shape[0]):
            for col in range(OBSTACLE_MAT.shape[1]):
                if OBSTACLE_MAT[row,col]:
                    rec_center = origin + np.array([UNIT * col, UNIT * row])
                    self.obstacle.append(
                        self.canvas.create_rectangle(
                            rec_center[0] - 15, rec_center[1] - 15,
                            rec_center[0] + 15, rec_center[1] + 15,
                            fill='black'
                        )
                    )

        # create oval
        oval_center = origin + np.array([UNIT * (MAZE_W-1), UNIT * (MAZE_H-1)])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

        # obstacle coordinate
        self.obstacle_coordinate = []
        for obs in self.obstacle:
            self.obstacle_coordinate.append(self.canvas.coords(obs))

    def reset(self):
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        next_coords = self.canvas.coords(self.rect) # next state
        old_dist = self.distance(self.canvas.coords(self.oval), s) # 计算前一个状态与终点
        new_dist = self.distance(self.canvas.coords(self.oval), next_coords)
        # reward function
        if new_dist == 0: # 到达终点
            reward = 100
            done = True
        elif(next_coords in self.obstacle_coordinate): # 碰到障碍物
            reward = -100
            done = True
        elif(old_dist == new_dist): # 到墙壁原地不动
            reward = -100
            done = True
        elif new_dist > old_dist:
            reward = -1
            done = False
        else:
            reward = 1
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        return s_, reward, done

    def render(self):
        time.sleep(0.01) # 每一步的更新时间
        self.update()

    def distance(self, obs1, obs2): # 计算两个位置之间的距离
        return ((obs1[0] - obs2[0])**2 + (obs1[1] - obs2[1])**2)**0.5

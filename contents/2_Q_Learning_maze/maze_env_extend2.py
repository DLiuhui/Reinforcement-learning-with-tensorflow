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
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width
OBSTACLE_MAT = np.array([[0,0,0,1,0,0],
                         [0,1,1,0,0,0],
                         [0,0,0,0,1,0],
                         [0,1,1,0,0,0],
                         [0,0,1,0,1,0],
                         [1,0,0,0,0,0]
                         ])

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.dist = 0

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
        # 初始化距离
        self.dist = self.distance(self.canvas.coords(self.oval), self.canvas.coords(self.rect))

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state
        # reward function
        # 每一步移动，如过离目标点更近，获得正奖励
        # 如果没有更近，获得负奖励
        # 碰到障碍物，最大负奖励
        # 到达重点，最大正奖励
        new_dist = self.distance(self.canvas.coords(self.oval), s_)
        if(s_ in self.obstacle_coordinate or 0 == self.distance(s,s_)):
            # 碰到障碍物 或者因为到达墙壁原地不动
            reward = -50
            done = True
            s_ = 'terminal'
        elif new_dist == 0: # 到达终点
            reward = 50
            done = True
            s_ = 'terminal'
        elif new_dist < self.dist: # 离终点更近
            reward = 1
            done = False
        else:
            reward = -1
            done = False
        self.dist = new_dist
        return s_, reward, done

    def render(self):
        time.sleep(0.05) # 每一步的更新时间
        self.update()

    def distance(self, obs1, obs2): # 计算两个位置之间的距离
        center1 = np.array([(obs1[0] + obs1[2])/2, (obs1[1] + obs1[3])/2])
        center2 = np.array([(obs2[0] + obs2[2]) / 2, (obs2[1] + obs2[3]) / 2])
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

import gym
import serial
import numpy as np
import math
import random
from numpy import sin, cos, pi

from gym import core, spaces

class continuumEnv(gym.Env):
    N = 6  # 骨骼的块数
    H = 6  # 一块骨骼的高度6mm
    h0 = 2.5  # 最初状态下两块骨骼间距2.5mm
    d = 15  # 四根cable圆的直径15mm
    L0 = 51  # 最初长度
    D = 20
    action = [-1, 1]
    timeStep = 0.02
    x_goal = 15
    y_goal = 15


    def __init__(self):

        self.viewer = None
        self.observation_space = [self.L0, self.L0, self.L0, self.L0]
        self.actions = [-1, 1]
        self.state = None
        self.L1 = None
        self.L2 = None
        self.L3 = None
        self.L4 = None
        self.reward = 0
        self.reset = 0
        self.done = 0
        self.on_goal = 0
        self.theta = 0
        self.phi = 0
        #self.seed()

        high = np.array([-1., -1.],dtype=np.float32)
        self.observation_space = spaces.Box(low=-0.1*high, high=0.1*high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def step(self, action1Idx, action2Idx, observation):
        '''
        ser = serial.Serial("COM3", 9600, timeout=0.5)
        if (action1Idx == 0) & (action2Idx == 0):
            action = "0"
        if (action1Idx == 0) & (action2Idx == 1):
            action = "1"
        if (action1Idx == 1) & (action2Idx == 0):
            action = "2"
        if (action1Idx == 1) & (action2Idx == 1):
            action = "3"
        ser.write(action.encode())
        '''
        #print(action)
        #print(aaa)
        self.done = 0
        x_goal = self.x_goal
        y_goal = self.y_goal
        state = np.array(self.state)
        x_error, y_error = state
###########得到cable长度##########

        L1, L2, L3, L4 = observation
        theta_max = 2 * math.atan(self.h0 / self.D)
        L_max = self.L0 + 2 * self.N * (self.d / 2 * sin(theta_max / 2) - self.h0 * (sin(theta_max / 4)) ** 2)

        if action1Idx == 0:
            action1 = -1
        else:
            action1 = 1
        self.L1 = L1 + action1 * self.timeStep
        self.L3 = 2*self.L0-self.L1

        if action2Idx == 0:
            action2 = -1
        else:
            action2 = 1
        self.L2 = L2 + action2 * self.timeStep
        self.L4 = L4 - action2 * self.timeStep
        if self.L1 >= L_max:
            self.L1 = L_max
            self.L3 = 2*self.L0 - self.L1
        if self.L2 >= L_max:
            self.L2 = L_max
            self.L4 = 2 * self.L0 - self.L2
        if self.L3 >= L_max:
            self.L3 = L_max
            self.L1 = 2*self.L0 - self.L3
        if self.L4 >= L_max:
            self.L4 = L_max
            self.L2 = 2*self.L0 - self.L4
        next_observation = np.array([self.L1, self.L2, self.L3, self.L4])

        a = self.L1
        b = self.L2
        c = self.L3
        d = self.L4

        n = self.N
        dd = self.d
##################得到next_state#################
        t = self.timeStep
        theta = 2 * math.asin(math.sqrt((a - c) ** 2 + (b - d) ** 2) / (2 * n * dd))

        if (a == c) & (b == d):
            phi = 0
            x = 0
            y = 0
        if (a == c) & (b < d):
            phi = -pi / 2
            x = 0
            y = (self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * sin(phi)
        if (a == c) & (b > d):
            phi = pi / 2
            x = 0
            y = (self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * sin(phi)
        if (a < c) & (b == d):
            phi = 0
            x = -(self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * cos(phi)
            y = 0
        if (a > c) & (b == d):
            phi = 0
            x = (self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * cos(phi)
            y = 0
        if (a > c) & (b > d):
            phi = math.atan2((b - d), (a - c))
            x = (self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * cos(phi)
            y = (self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * sin(phi)
        if (a > c) & (b < d):
            phi = -math.atan2((b - d), (a - c))
            x = abs((self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * cos(phi))
            y = -abs((self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * sin(phi))
        if (a < c) & (b < d):
            phi = math.atan2((b - d), (a - c))
            x = -abs((self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * cos(phi))
            y = -abs((self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * sin(phi))
        if (a < c) & (b > d):
            phi = math.atan2((b - d), (a - c))
            x = -abs((self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * cos(phi))
            y = abs((self.H + self.h0) * ((sin(self.N * theta / 2) ** 2) / sin(theta / 2)) * sin(phi))

        x_error = x_goal - x
        y_error = y_goal - y
        print(x,y)

        next_state = np.array([x_error, y_error])

        #更新状态奖励等
        self.check_goal()
        reward = self.reward
        done = self.done
        reset = self.reset

        if reset == 1:
            reset = 1
        else:
            reset = 0

        self.state = next_state
        self.done = done
        goal = self.x_goal
        #print(['theta:', state[0], 'theta_dot:', state[1], 'IsDone: ', is_done])
        return state, action1Idx, action2Idx, reward, next_state, done, reset, observation, next_observation, goal

    def check_goal(self):

        self.generate_reward()
        self.generate_reset()
        self.reward = -(abs(self.state[0]) + abs(self.state[1]))

        if (-0.1 < self.state[0] < 0.1) & (-0.1 < self.state[1] < 0.1):
            self.reward = self.reward + 1
            # self.done = 1

    def generate_reset(self):
        self.reset = 0

    def generate_reward(self):
        self.reward = 0



    def reset_code(self):

        self.x_goal = 0
        self.y_goal = 0
        self.state = np.array([self.x_goal, self.y_goal], dtype=float)
        self.on_goal = 0
        self.done = 0
        self.L1 = 51
        self.L2 = 51
        self.L3 = 51
        self.L4 = 51
        observation = np.array([self.L1, self.L2, self.L3, self.L4])

        return self.state, observation

    def reset_code1(self, last_state, i_count):
        self.x_goal = 0.1*i_count
        self.y_goal = 0.1*i_count
        x_last_error, y_last_error = last_state
        x_last_target = 0.1*(i_count-1)
        y_last_target = 0.1*(i_count-1)
        x_last = x_last_target-x_last_error
        y_last = y_last_target-y_last_error
        self.state = np.array([self.x_goal-x_last, self.y_goal-y_last], dtype=float)
        return self.state


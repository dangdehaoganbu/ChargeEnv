__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np
import random
import math
import gym
import itertools

from gym import spaces
from gym.error import DependencyNotInstalled


class ChargeEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        self.side_length = 1000
        self.N = 80   # 传感器的数量
        self.alpha = 36
        self.beta = 30
        self.x = random.uniform(0,1000)
        self.y = random.uniform(0,1000)
        self.pc = 0.9
        self.pr_th = 0.001
        self.eu = 5.6
        self.reset_flag = 1
        self.sensor_pos = None
        self.out_penalty = 1000 # 动作出界惩罚
        self.i_step = 0 # 记录动作步数

        self.action_space = spaces.Box(
            low=-1000, high=1000, shape=(1,2), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0, high=1000, shape=(1, self.N+2), dtype=np.float32)
        #self.observation_space = np.empty(shape=(1,self.N+2))

        # 创建传感器位置
        random_list = list(itertools.product(range(1, 1000), range(1, 1000)))
        self.sensor_pos = random.sample(random_list, 80)


    def step(self, action):
        action = action * 1000 #action的范围是-1~+1（加噪声），在这个地方逆归一化
        state = self.state
        next_state = state
        next_state[self.N], next_state[self.N+1] = state[self.N] + action[0][0], state[self.N+1] + action[0][1]
        self.i_step = self.i_step + 1

        # 出界判断
        out_flag = 0  # 出界标志，出界则引入惩罚
        for i in range(2):
            if next_state[self.N+i] > self.side_length:
                next_state[self.N+i] = 1000 # 淼哥的主意，想让这个方位尽量动起来
                out_flag = 1
            if next_state[self.N+i] < 0:
                next_state[self.N+i] = 0
                out_flag = 1
        # print('state:(',next_state[self.N],'，',next_state[self.N+1],')')
        d_move = math.sqrt(math.pow(action[0][0],2)+math.pow(action[0][1],2))
        cost_m = d_move*self.eu + out_flag * self.out_penalty

        # 计算充电状态损耗
        pos_x = self.state[self.N]
        pos_y = self.state[self.N+1]
        done = True  # False:未完成充电
        T_max = 0
        s_pos = self.sensor_pos
        d_th = math.sqrt(self.alpha * self.pc / self.pr_th) - self.beta # 充电的阈值距离
        c_total = 0 # 当前状态充电总量
        for i in range(self.N): # 把范围内的传感器充电置零，并获得最长充电时间
            d = math.sqrt(math.pow(s_pos[i][0] - pos_x, 2) + math.pow(s_pos[i][1] - pos_y, 2))
            if(d < d_th and state[i]>0):
                pr = self.alpha * self.pc / math.pow(d + self.beta, 2)
                t = state[i]/pr
                c_total = c_total + state[i]
                # print(c_total)
                if (t > T_max):
                    T_max = t
                next_state[i] = 0
        cost_p = self.pc * T_max

        # print('action:',action)
        reward = c_total * 10000 - (cost_p+cost_m) - self.i_step * 10
        # reward = - (cost_p+cost_m)
        # print(reward)
        c_total = 0
        for i in range(self.N):
            if(state[i]!=0):
                done = False
        return next_state, reward, done

# 问题先记录一下，关于python形参的问题
    def reset(self):
        # 创建传感器电量
        state_t = np.random.random(self.N)*2
        state_t = np.append(state_t, self.x)
        state_t = np.append(state_t, self.y)
        self.state = state_t
        self.i_step = 0

        return self.state

    def render(self, mode="human"):
        return None

    def close(self):
        return None



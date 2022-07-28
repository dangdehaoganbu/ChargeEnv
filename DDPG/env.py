#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:28:30
@LastEditor: John
LastEditTime: 2021-09-16 00:52:30
@Discription: 
@Environment: python 3.7.7
'''
import gym
import numpy as np
import random

class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''
    def action(self, action):
        low_bound   = self.action_space.low  # 这个core是什么？自己写的？怎么定义来的？
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound) # 这个映射？
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action

class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.2, decay_period=100000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_actions   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        return np.clip((action + ou_obs)*900, self.low, self.high) # 动作加上噪声后进行剪切

class GaussianNoise(object):
    '''高斯噪声
    '''
    def __init__(self, action_space, mu=0.0, sigma=10, max_sigma=0.3, min_sigma=0.2, decay_period=100000):
        self.mu           = mu # 高斯噪声的参数
        self.sigma        = sigma # 高斯噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_actions   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()

    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu

    def get_action(self, action, t=0):
        # action[0][0] = action[0][0] + random.gauss(self.mu, self.sigma)
        # action[0][1] = action[0][1] + random.gauss(self.mu, self.sigma) # 感觉这种类似遍历的方法有些笨笨的，有没有更优雅的方法
        [noi_0, noi_1] = [random.gauss(self.mu, self.sigma), random.gauss(self.mu, self.sigma)]
        # self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        # 等学习reward足够好了再来调整这个从前到后的更新
        return np.clip((action + [noi_0, noi_1]), -1, 1)  # 动作加上噪声后进行剪切
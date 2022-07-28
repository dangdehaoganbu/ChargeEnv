#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2022-06-09 19:04:44
@Discription: 
@Environment: python 3.7.7
'''
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        '''
        缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)   # 这句看似放屁，实则在给列表创建空位置，形成列表的框架
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)  # 和C不一样，python返回的是链表元素的个数，而非总体空间


# 简单的神经网络初始化内容
class Actor(nn.Module):  # nn是神经网络相关东西，这个模块的使用和具体构造还需在了解
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()  # 使用super方法则可以继承父类中的实例初始化函数
        """我建议就把这粗暴地理解神经网络的单层定义"""
        self.linear1 = nn.Linear(n_states, hidden_dim)  # 输入层
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.linear3 = nn.Linear(hidden_dim, n_actions)  # 输出层
        
        self.linear3.weight.data.uniform_(-init_w, init_w)  # 应该是机器学习里面的概念，权重怎么定？
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):   # 各层对应的激活函数，，每一层的最后的结果输出 ；前向传播
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        # 逆归一化的一次尝试
        # x = x*1000
        return x


class Critic(nn.Module):  # 策略网络，知道怎么直接调用就行了，可以再看一下飞桨，了解怎么直接用输出
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # 按维数1拼接
        # 第一眼看起来是x的连续赋值，但神经网络的运行和传递蕴含在了赋值右边的语句中。即x实现了前向传播的最终输出功能
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG:
    def __init__(self, n_states, n_actions, cfg):  #
        self.device = cfg.device  # 应该是将该神经网络放到配置的CPU/GPU上运行的意思，to device
        self.critic = Critic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        # 至此，四个网络设置完成

        # 复制参数到目标网络
        #                啥？这里的复制没懂，怎么复制？
        # 大概能明白这个参数复制的意思，具体的实现细节还需要看底层代码
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # import torch.optim as optim
        # 这是pytorch里专门用来进行参数更新参数优化的模块，
        # 配合pytorch的参数设定板块，传入学习率，能非常方便地对神经网络进行参数更新
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # 软更新参数 ，我一般称呼为tao
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 为什么此处要进行这样一个维度变换？
        action = self.actor(state)
        return action.detach().cpu().numpy()

    def update(self):
        if len(self.memory) < self.batch_size: # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device) # 这里的维度改变有些奇妙
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
       
        policy_loss = self.critic(state, self.actor(state))  # 对这个函数的返回的是策略网络的输出，对当前状态和动作进行打分。
        policy_loss = -policy_loss.mean()         # 策略网络优化部分 ，求均值？
        next_action = self.target_actor(next_state)  # 目标Q网络输出的下一步动作，为Q网络Loss更新准备
        target_value = self.target_critic(next_state, next_action.detach())  # detach的方法，将variable参数从网络中隔离开，不参与参数更新。
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        # torch.clamp 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

        action = action.squeeze(-2)
        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach()) # 时时刻刻防止更新？？？时时刻刻detach？为啥？
        """
        根据pytorch中backward（）函数的计算，
        当网络参量进行反馈时，梯度是累积计算而不是被替换，
        但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，
        因此需要对每个batch调用一遍zero_grad（）将参数梯度置0.
        """
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        # print('sadafaefaeffe', policy_loss.backward())
        self.actor_optimizer.step()
        # step(),更新所有参数,仍有疑问，优化器又不是网络，哪些参数被更新，思路和过程？
        # 感觉事情不对劲，optimizer也是一个神经网络吗？整个DDPG中不止4个网络？
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self,path):
        torch.save(self.actor.state_dict(), path+'checkpoint.pt')
        # state_dict()
        # 返回一个包含模块整体状态的字典。
        # 参数和持久性缓冲区（例如，运行平均数）都包括在内。
        # path是在task0里设置的存储的路径。
        # 但是也有疑问，我在服务器上得到模型参数，下载下来无法在本地上跑。

    def load(self,path):
        self.actor.load_state_dict(torch.load(path+'checkpoint.pt')) 
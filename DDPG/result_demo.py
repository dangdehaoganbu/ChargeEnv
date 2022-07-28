
import sys, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径， os模块的熟悉
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径sys.path

import datetime
import gym
import torch

from env import NormalizedActions, OUNoise
from ddpg import DDPG
from common.utils import save_results, make_dir
from common.utils import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    '''超参数
    '''

    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DDPG'  # 算法名称
        self.env_name = 'Pendulum-v1'  # 环境名称，gym新版本（约0.21.0之后）中Pendulum-v0改为Pendulum-v1
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 300  # 训练的回合数
        self.test_eps = 20  # 测试的回合数
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.99  # 折扣因子
        self.critic_lr = 1e-3  # 评论家网络的学习率 之前-3
        self.actor_lr = 1e-4  # 演员网络的学习率   之前-4
        self.memory_capacity = 8000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层维度
        self.soft_tau = 1e-2  # 软更新参数
        ################################################################################

        ################################# 保存结果相关参数 ################################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + '20220720-123614' + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


def env_agent_config(cfg, seed=1):
    env = NormalizedActions(gym.make(cfg.env_name))  # 装饰action噪声 ,gym.make 是建立环境
    env.seed(seed)  # 随机种子
    """
    space 是gym定义的环境空间参数。
    在 Gym 的仿真环境中，有运动空间 action_space 和观测空间 observation_space 两个指标，
    程序中被定义为 Space类型，用于描述有效的运动和观测的格式和范围。

    observation_space 是一个 Box 类型，box也是gym环境自定义的数据类型
    感觉box就是一个定义了 数据上下限、数据类型、维度的多位矩阵

    shape[0]代表什么意思？，动作指或者观测数据参数维度？
    """
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states, n_actions, cfg)
    return env, agent
    # 这样，咱的环境和智能体就安排好了

def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0  # 步长
        while not done:
            i_step += 1
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            """
            在 Gym 中，env.step() 会返回 4 个参数：

            观测 Observation (Object)：当前 step 执行后，环境的观测(类型为对象)。
            例如，从相机获取的像素点，机器人各个关节的角度或棋盘游戏当前的状态等；

            奖励 Reward (Float): 执行上一步动作(action)后，智体(agent)获得的奖励(浮点类型)，
            不同的环境中奖励值变化范围也不相同，但是强化学习的目标就是使得总奖励值最大；

            完成 Done (Boolen): 表示是否需要将环境重置 env.reset。大多数情况下，
            当 Done 为 True 时，就表明当前回合(episode)或者试验(tial)结束。
            例如当机器人摔倒或者掉出台面，就应当终止当前回合进行重置(reset);

            信息 Info (Dict): 针对调试过程的诊断信息。
            在标准的智体仿真评估当中不会使用到这个 info，具体用到的时候再说。
            """
            ep_reward += reward  # 每一个episode累计
            state = next_state
        print(state)
        # print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = Config()

    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=cfg.model_path)  # 我把服务器上的模型下载下来，无法运行，是哪里有问题吗？
    rewards, ma_rewards = test(cfg, env, agent)
    plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果


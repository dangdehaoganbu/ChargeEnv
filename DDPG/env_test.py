import gym

if __name__ == '__main__':
    env = gym.make('ChargeEnv-v0')
    state = env.reset()
    print(state)
    #env.step()

import numpy as np
import gym
import time


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.reset()
    
    done = False
    iteration = 0
    while(not done and iteration < 1000):
        iteration += 1
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        print(observation)
        time.sleep(0.1)
    
    
    env.close()



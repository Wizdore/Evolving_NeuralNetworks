import numpy as np
import gym

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.reset()
    for _ in range(200):
        env.render()
        env.step(1)

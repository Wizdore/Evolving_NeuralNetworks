import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(300):
    env.render()
    env.step(0) # take a random action
env.close()

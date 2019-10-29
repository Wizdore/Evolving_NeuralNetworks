'''
    key points:
        - https://gym.openai.com/docs/ OPen AI GYM
        - Pendulum starts upright
        - Prevent it from falling over
        - A rewar of +1 is provided for every timestep the pole remains upright
        - Applied a force of +1 or -1
        - If > 15 degrees from vertical or > 2.4 units from center , game ends
    Instructions:
        - Create a population of ANNS  who learn over multiple generations learn from its environment to better balance the pole
        - 3 layers size [n] [Random weights and biases]
            1.- Input [Random set of input parameters]
            2.- Hidden layer 
            3.- Output [Random set of output parameters]
        - Reading the observations from the environment and predicting their actions.
        - If prediction is Right retrain self ANN
        - If prediction is Wrong, stop training
        - 2 best ANN will creat 2 children [inherit part of wheight and bias]
        - Repeat until new ANN[N] = previous ANN[n]
'''
import gym
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(batch_size=1,max_iter=1, solver='sgd', activation='relu', learning_rate='invscaling', hidden_layer_sizes=hlayer_size, random_state=1)
partial_fit(np.array([env.observation_space.sample()]), np.array([env.action_space.sample()]), classes=np.arange(env.action_space.n)

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()




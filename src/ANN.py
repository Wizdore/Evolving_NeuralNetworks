import numpy as np
import time

# Fully connected 5-4-1 Neural network
# Code not **tested**, anyone wanna hand check to test the code? :D
# Very specific neural network code for just our case. number of nodes can be generelized a bit tho.. might do it later 

class ANN:
    def __init__(self, state):
        self.state = state                      # 29 size vector that holds weights and biases of whole network
        self.activation = lambda x: (2/(1+ np.exp(-0.5*x)))-1   # Slightly modified tanh Activation Function that should work better in our case
        self.generations = 20
        self.mutation_rate  = 0.001
        self.fitness = 0
    
    def set_hl_activation(self, activation):        # Optional function change the activation function
        self.activation = activation

    def get_output(self, inputs):
        hl_weight = self.state[0:20].reshape(5,4)       # First 20 elements are weights for input to hidden layer edges
        hl_bias = self.state[20:24]                     # Next 4 elements are for 4 nodes of hidden layer
        ol_weight = self.state[24:28].reshape(4,1)      # Next 4 elements are for 4 weights for hidden to output edges 
        ol_bias = self.state[28:29]                     # Next 1 element is the bias of the output layer

        inputs = self.activation(inputs)                # Normalizing the input values

        hl_result = (np.matmul(inputs, hl_weight) + hl_bias)        # calc from input to hidden layer (is linear activation good enough?!)
        ol_result = (np.matmul(hl_result, ol_weight) + ol_bias)     # calc from Hidden to output layer

        return ol_result        # returning raw result of the network

    def render(self, env):
        env.reset()
        iteration = 0
        done = False
        self.fitness = 0
        while(not done and iteration < np.iinfo(np.int32).max):
            iteration += 1
            env.render()
            observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
            self.fitness += reward
            #print(observation)
            time.sleep(0.01)
        print(self.fitness)

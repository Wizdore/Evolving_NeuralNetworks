import numpy as np
# Fully connected 5-4-1 Neural network
# Code not **tested**, anyone wanna hand check to test the code? :D
# Very specific neural network code for just our case. number of nodes can be generelized a bit tho.. might do it later 

class neural_net:
    def __init__(self, state):
        self.state = state                      # 29 size vector that holds weights and biases of whole network
        self.activation = lambda x: (2/(1+ np.exp(-0.5*x)))-1   # Slightly modified tanh Activation Function that should work better in our case
    
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

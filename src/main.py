import gym
import numpy as np

class ANNResult:
    """
    A class which resembles the result when simulating a neural network.
    """
    def __init__(self, state, inputs, lastStep):
        self.fitness = 0
        self.state = state
        self.inputs = None
        self.setInputs(inputs, lastStep)
    
    def getState(self):
        pass

    def setInputs(self, inputs, lastStep):
        self.inputs = np.concatenate([inputs, [lastStep]])

def crossover(parent1, parent2):
    """
    Parent1 and Parent2 are 2 states from 2 networks.
    The two parents are chosen from the top performers in the generation.
    Returns a tuple containing child 1 and child 2 which has mixed values from parent 1 and 2.
    """
    pivot = np.random.randint(0, len(parent1)) # Pick a random number.
    # Create children.
    child1 = parent1
    child2 = parent2
    # Crossover data.
    child1[:pivot] = parent2[:pivot]
    child2[:pivot] = parent1[:pivot]
    
    return (child1, child2)

def mutate_all(state, rate=0.001):
    """
    Go through each element in the state and check if that element should be mutated with
    probability 'rate'.
    """
    m = np.random.uniform(0.0,1.0,size=len(state))
    for idx, _ in enumerate(state):
        if rate >= m[idx]:
            state[idx] = np.random.uniform(-1,1)
    return state

def mutation_singular(state, rate=0.001):
    """
    Mutate single index (element) in state if mutation probability = TRUE.
    """
    if rate >= np.random.uniform(0,1):
        state[np.random.randint(0,len(state))] = np.random.uniform(-1,1)
    return state

def activationFunction(x):
    return ((2/(1+np.exp(-0.5*x)))-1)

def getOutput(inputs, state):
    hl_weight = state[0:20].reshape(5,4)       # First 20 elements are weights for input to hidden layer edges
    hl_bias = state[20:24]                     # Next 4 elements are for 4 nodes of hidden layer
    ol_weight = state[24:28].reshape(4,1)      # Next 4 elements are for 4 weights for hidden to output edges 
    ol_bias = state[-1]                     # Next 1 element is the bias of the output layer

    inputs = activationFunction(inputs)         # Normalizing the input values
    hl_result = (np.matmul(inputs, hl_weight) + hl_bias)        # calc from input to hidden layer (is linear activation good enough?!)
    ol_result = (np.matmul(hl_result, ol_weight) + ol_bias)     # calc from Hidden to output layer
    return ol_result        # returning raw result of the network

def simulate(env, ann, isFirstTime = False):
    """
    Run a network for some element in the population.
    """
    env.reset()
    done = False
    fitness = 0
    while(not done):
        env.render()
        observation, reward, done, _ = env.step(env.action_space.sample() if isFirstTime else getOutput(ann.inputs, ann.state))
        ann.setInputs(observation, 0) # predict next action based on observation?
        fitness += reward

    ann.fitness = fitness

if __name__ == "__main__":
    """
    Start execution of ANN Evolution here:
    - Create environment
    - Create an initial population
    - Run through X generations
    - For each generation run through Y population
    - For each population calculate fitness and rank the population for generation Z based on high-low fitness
    - Also keep track of the average score for each generation
    - Use the data to generate some data frame for use in excel (save as .csv)
    - WHAT ELSE ?
    """
    with gym.make('CartPole-v0') as env:
        population = [ANNResult(np.random.uniform(-1,1,29), env.observation_space.sample(), 0) for _ in range(40)] # Initial population
        for gen in range(15): # generation
            for ann in population:
                simulate(env,ann, (gen <= 0))
            
            population = sorted(population, key=lambda x: x.fitness, reverse=True) # Sort population based on best fitness.
            print("Best score for generation {} is {}.".format(gen+1, population[0].fitness))
            # mutate , GA stuff, make children, modify initial population ? scrap 50% worst entries
                        
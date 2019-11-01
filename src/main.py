import gym
import numpy as np

class ANNResult:
    """
    A class which resembles the result when simulating a neural network.
    """
    def __init__(self, state):
        self.fitness = 0
        self.state = state
        self.lastObservation = None
        self.lastOutput = None

def crossover(parent1, parent2):
    """
    Parent1 and Parent2 are 2 states from 2 networks.
    The two parents are chosen from the top performers in the generation.
    Returns a tuple containing child 1 and child 2 which has mixed values from parent 1 and 2.
    """
    pivot = np.random.randint(0, len(parent1.state)) # Pick a random number.
    # Create children.
    child1 = parent1.state
    child2 = parent2.state
    # Crossover data.
    child1[:pivot] = parent2.state[:pivot]
    child2[:pivot] = parent1.state[:pivot]
    
    return [ANNResult(child1), ANNResult(child2)]

def mutate_all(children, rate=0.001):
    """
    Go through each element in the state and check if that element should be mutated with
    probability 'rate'.
    """
    for child in children:
        m = np.random.uniform(0.0,1.0,size=len(child.state))
        for idx, _ in enumerate(child.state):
            if rate >= m[idx]:
                child.state[idx] = np.random.uniform(-1,1)

def mutation_singular(children, rate=0.001):
    """
    Mutate single index (element) in state if mutation probability = TRUE.
    """
    for child in children:
        if rate >= np.random.uniform(0,1):
            child.state[np.random.randint(0,len(child.state))] = np.random.uniform(-1,1)

def activationFunction(x):
    return ((2/(1+np.exp(-0.5*x)))-1)

def getOutput(ann):
    state = ann.state
    print(ann.lastObservation, ann.lastOutput)
    inputs = activationFunction(np.concatenate([ann.lastObservation, [ann.lastOutput]])) # Normalizing the input values

    hl_weight = state[0:20].reshape(5,4)       # First 20 elements are weights for input to hidden layer edges
    hl_bias = state[20:24]                     # Next 4 elements are for 4 nodes of hidden layer
    ol_weight = state[24:28].reshape(4,1)      # Next 4 elements are for 4 weights for hidden to output edges 
    ol_bias = state[-1]                     # Next 1 element is the bias of the output layer

    hl_result = (np.matmul(inputs, hl_weight) + hl_bias)        # calc from input to hidden layer (is linear activation good enough?!)
    ol_result = (np.matmul(hl_result, ol_weight) + ol_bias)     # calc from Hidden to output layer

    #return ol_result        # returning raw result of the network
    return (1 if ol_result >= 0 else 0)

def retainAndKeepBest(population):
    """
    Removes bad performers, crossovers and mutates best performers.
    Returns new population for the next gen.
    """
    population = sorted(population, key=lambda x: x.fitness, reverse=True) # Sort population based on best fitness.
    print("Best score for generation {} is {}.".format(gen+1, population[0].fitness))
    population = population[:(int(len(population)*0.5))] # keep top 50%
    size = len(population)
    for i in range(0, size, 2):
        parent1 = population[i]
        parent2 = population[i+1]
        children = crossover(parent1,parent2)
        mutate_all(children)
        population.extend(children)
    return population

def simulate(env, ann, isFirstTime = False):
    """
    Run a network for some element in the population.
    """
    env.reset()
    done = False
    while(not done):
        env.render()
        output = (env.action_space.sample() if isFirstTime else getOutput(ann))
        observation, reward, done, _ = env.step(output)
        ann.lastObservation = observation
        ann.lastOutput = output
        ann.fitness += reward

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
        population = [ANNResult(np.random.uniform(-1,1,29)) for _ in range(40)] # Initial population
        for gen in range(15): # generation
            for ann in population:
                simulate(env,ann,(gen <= 0))

            population = retainAndKeepBest(population)
            for p in population:
                print(p.state)
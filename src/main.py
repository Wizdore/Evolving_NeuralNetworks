import gym
import time
import ANN
import numpy as np

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

def mutate_all(state, rate):
    """
    Go through each element in the state and check if that element should be mutated with
    probability 'rate'.
    """
    m = np.random.uniform(0.0,1.0,size=len(state))
    for idx, _ in enumerate(state):
        if rate >= m[idx]:
            state[idx] = np.random.uniform(-1.0,1.0)
    return state

if __name__ == "__main__":
    with gym.make('CartPole-v0') as env:
        population = [ANN.ANN(np.random.uniform(0.0, 1.0, size=29)) for _ in range(50)] # Initial population
        for gen in range(10): # generation
            for network in population:
                network.render(env) # simulate

            population = sorted(population, key=lambda x: x.fitness, reverse=True) # Sort population based on best fitness.
            print("Best score for generation {} is {}.".format(gen+1, population[0].fitness))
            # mutate , GA stuff, make children, modify initial population ? scrap 50% worst entries
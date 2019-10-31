import gym
import ANN
import numpy as np

if __name__ == "__main__":
    with gym.make('CartPole-v0') as env:
        population = [ANN.ANN(np.random.uniform(-1.0, 1.0, size=29)) for _ in range(50)] # Initial population
        for gen in range(10): # generation
            for network in population:
                network.render(env) # simulate

            population = sorted(population, key=lambda x: x.fitness, reverse=True) # Sort population based on best fitness.
            print("Best score for generation {} is {}.".format(gen+1, population[0].fitness))
            # mutate , GA stuff, make children, modify initial population ? scrap 50% worst entries
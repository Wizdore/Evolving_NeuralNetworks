import gym
import time
import ANN
import numpy as np

if __name__ == "__main__":
    results = []
    with gym.make('CartPole-v0') as env:
        for _ in range(50):
            network = ANN.ANN(np.random.uniform(0.0, 1.0, size=29))
            network.render(env) # simulate
            results.append(network) # evaluate?

        results = sorted(results, key=lambda x: x.fitness, reverse=True)
        print("Best score:", results[0].fitness)

        # Run again with the 50% best?
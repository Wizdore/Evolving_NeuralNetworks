import gym
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ENV = gym.make('CartPole-v0')
mutation_rate = 0.005

class ANN:
    """
    Represents an Artificial Neural Network Class holding the instructions to evolve by crossover genetic algorithm
    in the manipulation for the carpole balancing problem
    """
    def __init__(self, state):
        self.state = state  # 25 size vector that holds weights and biases of whole network
        self.activation = lambda x: (2 / (1 + np.exp(-0.5 * x))) - 1  # Slightly modified tanh Activation
        self.fitness = 0

    def set_hl_activation(self, activation):  # Optional function change the activation function
        self.activation = activation

    def get_output(self, inputs):
        """
        Returns the actual output,
        an activation function can be used at a later point to return the action, 
        which can take the value 0 or 1.
        """
        hl_weight = self.state[0:16].reshape(4, 4)      # First 16 elements are weights for input to hidden layer edges
        hl_bias = self.state[16:20]                     # Next 4 elements are biases for 4 nodes of hidden layer
        ol_weight = self.state[20:24].reshape(4, 1)     # Next 4 elements are for 4 weights for hidden to output edges
        ol_bias = self.state[-1]                        # Next 1 element is the bias of the output layer

        inputs = self.activation(inputs)  # Normalizing the input values

        hl_result = (np.matmul(inputs,
                               hl_weight) + hl_bias)  # calc from input to hidden layer (is linear activation good enough?!)
        ol_result = (np.matmul(hl_result, ol_weight)+ol_bias)  # calc from Hidden to output layer
        
        return (1 if  ol_result>= 0 else 0)   # returning result of the network after binary step activation

def crossover(parent1, parent2):
    """
    Parent1 and Parent2 are 2 states from 2 networks.
    The two parents are chosen from the top performers in the generation.
    Returns a tuple containing child 1 and child 2 which has mixed values from parent 1 and 2.
    Flips a coin for each element in the state and swaps the states between parents or keeps them as is.
    """
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    m = np.random.uniform(0.0, 1.0, size=len(parent1))
    for idx, _ in enumerate(parent1):
        if m[idx] > 0.5:
            child1[idx] = parent2[idx]
            child2[idx] = parent1[idx]
        else:
            child1[idx] = parent1[idx]
            child2[idx] = parent2[idx]

    return child1, child2

def mutate_all_slightly(state, max_mutation_amount=0.5):
    """
    Go through each element in the state and check if that element should be mutated with
    probability X, if a random uniform value between [0,1] is less or equal to this X, mutate that element.
    """
    m = np.random.uniform(0.0, 1.0, size=len(state))
    for idx, _ in enumerate(state):
        if mutation_rate >= m[idx]:
            mutval = state[idx] + np.random.uniform(-max_mutation_amount, max_mutation_amount)
            mutval = np.tanh(mutval)
            state[idx] = mutval

def retainAndKeepBest(population, percent=0.5):
    """
    Removes bad performers, then does a crossover and potentially mutates some of the new children.
    Returns new population for the next generation.
    """
    population = sorted(population, key=lambda x: x.fitness, reverse=True)  # Sort population based on best fitness.
    population = population[:(int(len(population) * percent))]  # keep top 50%
    size = len(population)
    for i in range(0, size, 2):
        population[i].fitness = 0
        population[i + 1].fitness = 0
        parent1_dna = population[i].state
        parent2_dna = population[i + 1].state
        child1_dna, child2_dna = crossover(parent1_dna, parent2_dna)

        mutate_all_slightly(child1_dna)
        mutate_all_slightly(child2_dna)

        child1 = ANN(child1_dna)
        child2 = ANN(child2_dna)

        population.extend([child1, child2])
    return population

def test_population(population):
    """
    Test every element in the population, record the fitnes for every element.
    Predict the action (output) based on previous observations.
    """
    for child in population:
        observation = ENV.reset()
        done = False
        while not done:
            action = child.get_output(observation)
            observation, reward, done, _ = ENV.step(action)
            child.fitness += reward

gens = []
worst_scores = []
median_scores = []
best_scores = []

best_agents = []
median_agents = []
worst_agents = []

def record_population_score(population, gen_number):
    """
    Retrieve the highest, median and worst fitness from the population of generation X.
    """
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    gens.append(gen_number)
    best_agents.append(population[0])
    median_agents.append(population[int(len(population) / 2)])
    worst_agents.append(population[-1])

    best_scores.append(population[0].fitness)
    median_scores.append(population[int(len(population) / 2)].fitness)
    worst_scores.append(population[-1].fitness)

    print(f"Gen {gen_number}: Worst {worst_scores[-1]}\tMedian {median_scores[-1]},\tBest {best_scores[-1]}")
    test_agent(population[0])  # Rendering the best Agent

def animate(i):
    """
    Animates a live graph, graph is updated at a given interval.
    """
    plt.cla()
    plt.plot(gens, worst_scores, label='Worst')
    plt.plot(gens, median_scores, label='Median')
    plt.plot(gens, best_scores, label='Best')
    plt.xlabel("Generations")
    plt.ylabel("Fitness score")
    plt.title("Scores of Worst/Median/Best agent over generations")
    plt.legend(loc='upper left')
    plt.tight_layout()

def test_agent(agent):
    """
    Run a single agent in a different environment.
    """
    print('Showing agent with fitness: {}'.format(agent.fitness))
    observation = ENV.reset()
    done = False
    while not done:
        ENV.render()
        action = agent.get_output(observation)
        observation, _, done, _ = ENV.step(action)
    time.sleep(0.2)

def evolve(n_generations, initialpop_size):
    """
    Evolves a population of X neural networks over a given number of generations.
    For each new generation, the 50% best performers are kept and new children are bred from these parents.
    The worst, best and avg scores are shown after each generation.
    """
    global ENV
    population = [ANN(np.random.uniform(-1, 1, 25)) for _ in range(initialpop_size)]  # Initial population
    for gennum in range(n_generations):
        test_population(population)
        record_population_score(population, gennum + 1)
        population = retainAndKeepBest(population)
    ENV.close()
    ENV = None    

if __name__ == "__main__":
    """
    Executes the ANN-evolution algorithm with the given parameters on its on thread.
    The main thread launches a live plot which will visualize the performance for each generation.
    """
    generations_to_run = 40
    initial_population_size = 48

    evolution_thread = threading.Thread(target=evolve, args=(generations_to_run, initial_population_size))
    evolution_thread.start()

    graphAnimation = FuncAnimation(plt.gcf(), animate, interval=500)
    plt.tight_layout()
    plt.show()
    evolution_thread.join()
    print(best_agents[-1].state)
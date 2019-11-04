import gym
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ENV = gym.make('CartPole-v0')
mutation_rate = 0.005


class ANN:
    def __init__(self, state):
        self.state = state  # 24 size vector that holds weights and biases of whole network
        self.activation = lambda x: (2 / (1 + np.exp(-0.5 * x))) - 1  # Slightly modified tanh Activation
        self.fitness = 0

    def set_hl_activation(self, activation):  # Optional function change the activation function
        self.activation = activation

    def get_output(self, inputs):
        hl_weight = self.state[0:16].reshape(4, 4)  # First 20 elements are weights for input to hidden layer edges
        hl_bias = self.state[16:20]  # Next 4 elements are for 4 nodes of hidden layer
        ol_weight = self.state[20:24].reshape(4, 1)  # Next 4 elements are for 4 weights for hidden to output edges
        # ol_bias = self.state[-1]                     # Next 1 element is the bias of the output layer

        inputs = self.activation(inputs)  # Normalizing the input values

        hl_result = (np.matmul(inputs,
                               hl_weight) + hl_bias)  # calc from input to hidden layer (is linear activation good enough?!)
        ol_result = (np.matmul(hl_result, ol_weight))  # calc from Hidden to output layer

        return ol_result  # returning raw result of the network


def crossover(parent1, parent2):
    """
    Parent1 and Parent2 are 2 states from 2 networks.
    The two parents are chosen from the top performers in the generation.
    Returns a tuple containing child 1 and child 2 which has mixed values from parent 1 and 2.
    """

    parent1_weight = parent1[:20]
    parent2_weight = parent2[:20]
    parent1_bias = parent1[20:]
    parent2_bias = parent2[20:]

    pivotMin_weight = np.random.randint(0, len(parent1_weight))
    pivotMax_weight = np.random.randint(pivotMin_weight, len(parent1_weight))

    pivotMin_bias = np.random.randint(0, len(parent1_bias))
    pivotMax_bias = np.random.randint(pivotMin_bias, len(parent2_bias))

    # Create children.
    child1 = np.hstack((parent1_weight, parent1_bias))
    child2 = np.hstack((parent2_weight, parent2_bias))
    # Crossover data.
    child1[pivotMin_weight:pivotMax_weight] = parent2[pivotMin_weight:pivotMax_weight]
    child1[pivotMin_bias:pivotMax_bias] = parent2[pivotMin_bias:pivotMax_bias]

    child2[pivotMin_weight:pivotMax_weight] = parent1[pivotMin_weight:pivotMax_weight]
    child2[pivotMin_bias:pivotMax_bias] = parent1[pivotMin_bias:pivotMax_bias]

    return child1, child2


def mutate_all(state):
    """
    Go through each element in the state and check if that element should be mutated with
    probability 'rate'.
    """
    m = np.random.uniform(0.0, 1.0, size=len(state))
    for idx, _ in enumerate(state):
        if mutation_rate >= m[idx]:
            state[idx] = np.random.uniform(-1, 1)


def mutate_all_slightly(state, max_mutation_amount=0.5):
    """
    Go through each element in the state and check if that element should be mutated with
    probability 'rate'.
    """
    m = np.random.uniform(0.0, 1.0, size=len(state))
    for idx, _ in enumerate(state):
        if mutation_rate >= m[idx]:
            mutval = state[idx] + np.random.uniform(-max_mutation_amount, max_mutation_amount)
            mutval = np.tanh(mutval)
            state[idx] = mutval


def mutation_single(state):
    """
    Mutate single index (element) in state if mutation probability = TRUE.
    """
    if mutation_rate >= np.random.uniform(0, 1):
        state[np.random.randint(0, len(state))] = np.random.uniform(-1, 1)


def retainAndKeepBest(population, percent=0.5):
    """
    Removes bad performers, crossovers and mutates best performers.
    Returns new population for the next gen.
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
    Run a network for some element in the population.
    """
    for child in population:
        observation = ENV.reset()  # random inputs
        done = False
        while not done:
            # ENV.render()
            action = (1 if child.get_output(observation) >= 0 else 0)
            observation, reward, done, _ = ENV.step(action)
            child.fitness += (reward + ((2 / (1 + np.exp(-0.1 * abs(observation[1])))) - 1)/2) 


gens = []
worst_scores = []
median_scores = []
best_scores = []

best_agents = []
median_agents = []
worst_agents = []


def record_population_score(population, gen_number):
    #  TODO: We can store these in pandas dataframe too
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    gens.append(gen_number)
    best_agents.append(population[0])
    median_agents.append(population[int(len(population) / 2)])
    worst_agents.append(population[-1])

    best_scores.append(population[0].fitness)
    median_scores.append(population[int(len(population) / 2)].fitness)
    worst_scores.append(population[-1].fitness)

    print(f"Gen {gen_number}: Worst {worst_scores[-1]}\tMedian {median_scores[-1]},\tBest {best_scores[-1]}")
    test_agent(population[0])  ## Rendering the best Agent


def animate(i):
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
        action = (1 if agent.get_output(observation) >= 0 else 0)
        observation, reward, done, _ = ENV.step(action)
    time.sleep(0.2)
    return


def evolve(n_generations, initialpop_size):
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
    population = [ANN(np.random.uniform(-1, 1, 24)) for _ in range(initialpop_size)]  # Initial population
    for gennum in range(n_generations):
        test_population(population)
        record_population_score(population, gennum + 1)
        population = retainAndKeepBest(population)


if __name__ == "__main__":
    generations_to_run = 50
    initial_population_size = 60
    mutation_rate = 0.005

    evolution_thread = threading.Thread(target=evolve, args=(generations_to_run, initial_population_size))
    evolution_thread.start()

    ani = FuncAnimation(plt.gcf(), animate, interval=500)
    plt.tight_layout()
    plt.show()
    ENV.close()
    evolution_thread.join()
    print(best_agents[-1].state)
    exit()

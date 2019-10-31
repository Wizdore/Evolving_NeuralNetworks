import numpy as np

def mutation_old1(state, rate=0.01):
    # create array with 29 elements, which are either 0 or 1
    a = np.random.binomial(1, rate, 29)
    for idx, elem in enumerate(a):
        if elem == 1:
            state[idx] = np.random.uniform(-1,1)
    return state

def mutation_old(state, rate=0.01):
    # This function works up to a mutation rate of 0.0345
    if np.random.binomial(1, rate*len(state), 1) == 1:
        state[np.random.randint(0,len(state))] = np.random.uniform(-1,1)
    return state

def mutation(state, rate_state=0.01):
    # This function works up to a mutation rate of 0.0345
    if np.random.uniform(0,1) <= rate_state:
        state[np.random.randint(0,len(state))] = np.random.uniform(-1,1)
    return state
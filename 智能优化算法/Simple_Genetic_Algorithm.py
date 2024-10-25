import numpy as np
import random
import matplotlib.pyplot as plt

def elite_selection(fitness):
    return fitness.argsort()[-2:][::-1]  

def crossover(parent1, parent2, crossover_rate):
    # children are copies of parents by default
    child1, child2 = parent1.copy(), parent2.copy()  
    # check for recombination
    if random.random() < crossover_rate:
        # select crossover point that is not on the end of the string
        pt = random.randint(1, len(parent1)-2)
        # perform crossover    
        child1 = np.concatenate((parent1[:pt], parent2[pt:]))
        child2 = np.concatenate((parent2[:pt], parent1[pt:]))
    return [child1, child2]


def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        # check for a mutation
        if random.random() < mutation_rate:
            # flip the bit
            individual[i] = 1 - individual[i]
    return individual

def simple_GA(pop, crossover_rate=.5, mutation_rate=.05):
    fitness = np.sum(pop,axis=1) 
    parents = elite_selection(fitness)
    children = np.zeros((population,genes))  
    for i in range(population):
        offspring = crossover(pop[parents[0]],pop[parents[1]], crossover_rate)
        children[i] = mutation(offspring[0],mutation_rate)  
    return children


#initial population
population = 100
genes = 100
generations = 100

pop = np.random.randint(0,2, size=(population,genes))

simple_GA(pop)
import numpy as np
import random
from . import utils

def remainder_stochastic_sampling(fitnessed, population):
    intermediate_population = np.array([])
    for fitness, genotype in zip(fitnessed, population):
        # Copy objects with >1 fitness (mod 1) times to intermediate
        while fitness >= 1:
            intermediate_population = np.append(intermediate_population, genotype)
            fitness -= 1
        # Calculate probability of adding to intermediate and add if event occurs
        if utils.compute_chance(fitness):
            intermediate_population = np.append(intermediate_population, genotype)
    return intermediate_population

def stohastic_universal_sampling(fitnessed, population):
    # print(fitnessed)
    # print()
    # print(population)
    ftns_n = len(fitnessed)
    n = len(population)
    dist = ftns_n / n
    start = random.randint(1, dist)
    pointers = [start + i*dist for i in range(n)]
    # print(ftns_n, n, dist, start, pointers)
    keep = np.array([])
    for pointer in pointers:
        i = 0
        while np.sum(fitnessed[:i]) < pointer:
            i += 1
        keep = np.append([population[i]], keep)
    return keep

def proportional_selection(population, points_map, f, dim=1, selector=stohastic_universal_sampling,
                           evaluated=None):
    fitnesses = utils.fitness(population, points_map, f, dim, evaluated)
    intermediate_population = selector(
        np.asarray(fitnesses[:, -1], dtype=np.float128),
        fitnesses[:, 0],
    )
    return intermediate_population

def one_point_crossover(intermediate_population):
    n = len(intermediate_population[0])
    new_interm_population = np.array([])
    for i in range(len(intermediate_population)//2):
        crossover_point = random.randint(1, n-1)
        child1 = intermediate_population[2 * i][:crossover_point] + intermediate_population[2 * i + 1][crossover_point:]
        child2 = intermediate_population[2 * i + 1][:crossover_point] + intermediate_population[2 * i][crossover_point:]
        new_interm_population = np.append([child1, child2], new_interm_population)
    return new_interm_population

def mutation(population, p=0.1):
    flip_bit = lambda x, i: x[:i] + ('1' if x[i] == '0' else '0') + x[i+1:]
    mutate = lambda x: flip_bit(x, random.randint(0, len(x)-1)) if utils.compute_chance(p) else x
    return np.vectorize(mutate)(population)

if __name__ == "__main__":
    genotype_length = 8
    gray_codes = utils.get_n_grey(genotype_length)
    points_map = utils.map_on_search_area(gray_codes, [-5, 5])
    population = utils.build_random_population(50, points_map)

    f = lambda x: -x**2

    i = 0
    while population.size > 5:
        population = proportional_selection(
            population,
            points_map,
            f,
            selector=remainder_stochastic_sampling
        )
        population = one_point_crossover(population)
        population = mutation(population)
        np.random.shuffle(population)
        print("{}: number of genotypes {}".format(i, population.size))
        i+=1

    print(utils.evaluate_population(population, points_map, f))
    print(utils.fitness(population, points_map, f))
    best = utils.best_genotype(population, points_map, f)
    print("best genotype {} ar point {} with value {}".format(best[0], best[1], best[2]))

    # print(selected)


import numpy as np
import random
import time
import numba as nb

def get_n_grey(n: int) -> list :
    grays = []
    for i in range(0, 1<<n):
        gray=i^(i>>1)
        grays.append("{0:0{1}b}".format(gray,n))
    return grays

def dstack_product(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

def map_on_search_area(gray_codes: list, search_area: list, dim=1) -> np.ndarray :
    points = np.linspace(search_area[0], search_area[1], len(gray_codes))
    if dim == 1:
        return np.column_stack((gray_codes, points))
    elif dim == 2:
        all_points = [points] * dim
        producted_points = dstack_product(*all_points)
        producted_gray_codes = dstack_product(*([[gray_codes]]*dim))
        concate = (producted_gray_codes[i, 0] + producted_gray_codes[i, 1] for i in range(producted_gray_codes.shape[0]))
        concated_codes = np.fromiter(concate, dtype='U'+str(len(gray_codes[0])*2))
        return np.column_stack((concated_codes, producted_points))

def evaluate_population(population: np.ndarray, points_map: np.ndarray, f, dim=1) -> np.ndarray :
    real_points_mp = points_map[np.isin(points_map[:, 0], population)]
    real_points = np.asarray(real_points_mp[:, 1:], dtype=np.float128)
    values = np.apply_along_axis(f, 1, real_points)
    evaled = np.column_stack((real_points_mp, values))
    return evaled

def build_random_population(size: int, points_map: np.ndarray) -> np.ndarray :
    population = np.array([])
    while population.size <= size:
        genotype = points_map[random.randint(0, points_map.shape[0]-1), 0]
        population = np.append([genotype], population)
    return population

def compute_chance(prob: float) -> bool:
    w = random.random()
    return w <= prob

def fitness(population, points_map, f, dim=1, evaluated=None):
    evaluation = evaluate_population(population, points_map, f, dim) if type(evaluated) == type(None) else evaluated
    values = np.asarray(evaluation[:, -1], dtype=np.float128)
    # fitnesses = values - np.min(values)
    # fitnesses = fitnesses / np.average(fitnesses) + 0.1
    fitnesses = values / np.average(values)
    # fitnesses = fitnesses - np.min(fitnesses)
    return np.column_stack((evaluation, fitnesses))


def population_converged(population, f, eps=0.1, target='min'):
    pass

def best_genotype(population, points_map, f, dim=1):
    evaled = evaluate_population(population, points_map, f, dim)
    real_points = np.asarray(evaled[:, -1], dtype=np.float128)
    best_idx = np.argmax(real_points)
    return evaled[best_idx]

def generate_codes_and_population(gen_length, search_area, n_population, dim):
    print("Generating codes...")
    gray_codes = get_n_grey(gen_length)
    points_map = map_on_search_area(gray_codes, search_area, dim)
    print("Initializing first population...")
    population = build_random_population(n_population, points_map)
    return points_map, population


if __name__ == "__main__":
    genotype_length = 10
    population_size = 10

    # f = lambda x: x**100
    # dim = 1
    f = lambda x: x[0]**2 + x[1]**2
    dim = 2

    gray_codes = get_n_grey(genotype_length)
    points_map = map_on_search_area(gray_codes, [-1, 1], dim=dim)
    print("Map size", points_map.shape)
    print(points_map)
    population = build_random_population(population_size, points_map)
    print("Built population")
    # population = list(population)
    # print(population)

    evaled = evaluate_population(population, points_map, f, dim=dim)
    # print(evaled)
    # fitnesses = fitness(population, points_map, f, dim=dim)
    # print(fitnesses)


import numpy as np
from . import utils
from . import operators

def CGA(f, search_area, dim=1, gen_length=16, n_population=50, iterations=100,
        selection=operators.proportional_selection,
        selector=operators.remainder_stochastic_sampling,
        ):
    points_map, population = utils.generate_codes_and_population(gen_length, search_area, n_population, dim)

    population_evaluation = utils.evaluate_population(population, points_map, f, dim)
    populations = [population_evaluation]
    i = 0
    while population.size > 5 and i < iterations:
        if selection == operators.proportional_selection:
            population = selection(
                population,
                points_map,
                f,
                selector=selector,
                dim=dim,
                evaluated=population_evaluation,
            )
        else:
            population = selection(
                population,
                points_map,
                f,
                evaluated=population_evaluation,
            )

        population = operators.one_point_crossover(population)
        population = operators.mutation(population)
        np.random.shuffle(population)

        population_evaluation = utils.evaluate_population(population, points_map, f, dim)
        populations.append(population_evaluation)
        fitnesses = utils.fitness(population, points_map, f, dim, population_evaluation)
        ftns = np.asarray(fitnesses[:, -1], np.float128)
        print("generation {}: number of gens {}, avg fitness {}".format(i, population.size, np.average(ftns)))
        i += 1

    best = utils.best_genotype(population, points_map, f, dim)
    return best, populations


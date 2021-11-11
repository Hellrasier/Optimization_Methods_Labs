from test_functions import *
from genetical_algorithms import genetical_algorithms

area = [-4, 4]
best, populations = genetical_algorithms.CGA(
    lambda x: -rastrigin(x),
    area,
    dim=2,
    gen_length=12,
)

print("best genotype {} ar point {} with value {}".format(best[0], best[1:-1], best[-1]))

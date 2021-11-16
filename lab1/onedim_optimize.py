from scipy.misc import derivative
import numpy as np
from genetical_algorithms import utils, operators
import math as m
from scipy.optimize import minimize_scalar
def der(f, x, n):
    return derivative(f, x, dx=np.float64(1e-8), n=n)

# def der2(f, x):
#     eps = np.float64(1e-6)
#     der1 = der(f, x, 1)
#     x_dx = x + eps
#     der2 = der(f, x_dx, 1)
#     return np.divide(der2 - der1, eps)

def brent_optimize(f, a, b, epsilon):
    res = minimize_scalar(f)
    return res.x, res.nfev

def getFibbonachies(epsilon):
    fibbs = [] 
    fibbs.append(1)
    fibbs.append(1)
    while(fibbs[len(fibbs) - 1] < 1/epsilon):
        f2 = fibbs[len(fibbs) - 1]
        f1 = fibbs[len(fibbs) - 2]
        fibbs.append(f1 + f2)
    return fibbs
    
def fibbonaci_method(f, a, b, epsilon):
    F = getFibbonachies(epsilon/1000)
    N = len(F) - 2
    l = b - a
    delta = epsilon / 100
    x2 = a + F[N]/F[N+1] * l + (-1)**(N+1)*2*delta/F[N+1]
    x1 = a + F[N-1]/F[N+1] * l
    f_x1, f_x2 = f(x1), f(x2)
    k = 1
#     print(a, b)
    a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2)
#     print(a, b)
    for i in range(2, N+1):
        y = a + F[N-i]/F[N+1] * l + (-1)**(N+1)*2*delta/F[N+1]
        if x == y: y = b - F[N-i]/F[N+1] * l + (-1)**(N+1)*2*delta/F[N+1]
        if x < y:
            x1, x2 = x, y
            f_x1, f_x2 = f_x, f(y)
            k += 1
        else: 
            x1, x2 = y, x
            f_x1, f_x2 = f(y), f_x
            k += 1
        a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2) 
#         print(a, b)
    if f((a+b)/2) > f(b):
            return b, k+1
    else:
        return (a+b)/2, k+1


def upgraded_newton(f, a, b, epsilon):
    der1 = der(f, a, 1)
    sec_der = der(f, a, 2)
#     print("sec der", sec_der)
    if sec_der <= epsilon:
#         print("Starting fibbonaci")
        x, k = fibbonaci_method(f, a, b, epsilon)
        return x, k
    x0 = a
    f_ev = 4
    x1 = x0 - np.divide(der1, sec_der)
#     print(x1)
    der2 = der(f, x1, 1)
    f_ev += 2
    while(m.fabs(x0 - x1) > epsilon):
        if np.absolute(der2 - der1) < epsilon:
            x, k = fibbonaci_method(f, a, b, epsilon)
            f_ev += k
            return x, f_ev
        x2 = x1 - np.divide(x1 - x0, der2 - der1) * der2
#         print(x2)
        x0 = x1
        x1 = x2
        der1 = der2
        der2 = der(f, x1, 1)
        f_ev += 2
#     if x1 > b:
#         x1 = b
    return x1, f_ev          

def find_parabola_minimum(x1, x2, x3, f1, f2, f3):
    a = f1/((x1-x2)*(x1-x3)) + f2/((x2-x1)*(x2-x3)) + f3/((x3-x1)*(x3-x2))
    b = (f1 * (x2 + x3))/((x1-x2)*(x3-x1)) + (f2 * (x1 + x3))/((x2-x1)*(x3-x2)) + (f3 * (x1 + x2))/((x3-x1)*(x2-x3))
    return -b/(2*a)

def cut_section_procedure(a, x1, x2, b, f_x1, f_x2):
    if f_x1 <= f_x2:
        return a, x2, x1, f_x1
    else:
        return x1, b, x2, f_x2

def quadratic_approx(f, a, b, epsilon):
    x1, x3 = a, b
    x2 = a + (b - a) * 1/1.618033988
    f1, f2, f3 = f(x1), f(x2), f(x3)
    k = 1
    print(x1, x2, x3)
    while(True):
        xm = find_parabola_minimum(x1, x2, x3, f1, f2, f3)
        fm = f(xm)
        k += 1
        if xm < x3 and xm >= x2 and fm <= f2:
            x1, x2 = x2, xm
            f1, f2 = f2, fm
        elif xm < x3 and xm >= x2 and fm > f2:
            x3 = xm
            f3 = fm
        elif xm <= x2 and xm > x1 and fm <= f2:
            x2, x3 = xm, x2
            f2, f3 = fm, f2
        elif xm <= x2 and xm > x1 and fm > f2:
            x1 = xm
            f1 = fm
        print(x1, x2, x3, xm, fm)
        if(x3-x2 < epsilon or x2-x1 < epsilon): break
    return xm, k+2

def newton_modified(f, a, b, epsilon):
    der1 = der(f, a, 1)
    derb = der(f, b, 1)
    x0 = a
    x1 = np.divide(a * derb - b*der1, derb - der1)
    der2 = der(f, x1, 1)
    k = 0
    while(m.fabs(x0 - x1) > epsilon):
        x2 = x1 - np.divide(x1 - x0, der2 - der1) * der2
        x0 = x1
        x1 = x2
        der1 = der2
        der2 = der(f, x1, 1)
        k += 1
    return x1, k

def middle_point_method(f, a, b, eps=1e-8):
    condition = lambda a, b: b - a < eps
    x = None
    k = None
    iteration = 0    
    while not condition(a, b) and k != 0:
        x = (a + b)/2
        k = derivative(f, x, dx=eps)        
        if k > 0:
            a, b = a, x
        elif k < 0:
            a, b = x, b
        iteration += 1
        
    return x, iteration

def find_mu(x1, x2, f1, f2, df1, df2):
    z = df1 + df2 - 3*np.divide(f2-f1, x2-x1)
    w = np.sqrt(z**2 - df1*df2)
    mu = np.divide(w + z - df1, 2*w - df1 + df2) 
    return mu
    
def qubic_approx(f, a, b, epsilon):
    x1, x2 = a, b
    f1, f2 = f(a), f(b)
    df1, df2 = der(f, a, 1), der(f, b, 1)
#     print("a b ders", df1, df2, df1*df2)
    if(df1*df2 > 0):
        return b, 1
    mu = find_mu(a, b, f(a), f(b), df1, df2)
    xm = x1 + mu*(x2 - x1)
    dfm = der(f, xm, 1)
    fm = f(xm)
#     print(x1, xm, x2, fm)
    if df1*dfm < 0:
        x2 = xm
        f2 = fm
        df2 = dfm
    else:
        x1 = xm
        f1 = fm
        df1 = dfm
    k = 1
    while(np.absolute(dfm) > epsilon and x2 - x1 > epsilon/100):
#         print("x2 - x1", x2 - x1)
        mu = find_mu(x1, x2, f1, f2, df1, df2)
        xm = x1 + mu*(x2 - x1)
        dfm = der(f, xm, 1)
        fm = f(xm)
#         print(x1, xm, x2, fm)
        if df1*dfm < 0:
            x2 = xm
            f2 = fm
            df2 = dfm
        else:
            x1 = xm
            f1 = fm
            df1 = dfm
        k += 1
    return xm, k


def genetical_algorithm(f, a, b, epsilon):
    search_area = [a, b]
    gen_length = int(np.ceil(np.log2(np.divide(search_area[1] - search_area[0], epsilon))))
    dim = 1
    n_population=50 
    iterations=100
    selection=operators.proportional_selection
    selector=operators.remainder_stochastic_sampling
    points_map, population = utils.generate_codes_and_population(gen_length, search_area, n_population, dim)
    population_evaluation = utils.evaluate_population(population, points_map, f, dim)
    f_ev = population.shape[0]
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
        f_ev += population.shape[0]
        populations.append(population_evaluation)
        fitnesses = utils.fitness(population, points_map, f, dim, population_evaluation)
        ftns = np.asarray(fitnesses[:, -1], np.float128)
        print("generation {}: number of gens {}, avg fitness {}".format(i, population.size, np.average(ftns)))
        i += 1
    best = utils.best_genotype(population, points_map, f, dim)
    return np.float64(best[1]), f_ev




# def one_dim_optimizer(f, a, b, eps):
#     K = 0
#     a, b, k = fibbonaci_method(f, a, b, eps*100)
#     print("Fibbonaci finished with a=", a, "b=", b)
#     K += k
#     x, k = qubic_approx(f, a, b, eps)
#     print("Qubic finished with x=", x)
#     K = k
#     return x, K


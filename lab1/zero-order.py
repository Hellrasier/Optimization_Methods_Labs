import numpy as np
from numpy.linalg import norm
import matplotlib.animation as pltanimation
from test_functions import *
from animations import AnimateSimplex, Animate3D, load_animation
from onedim_optimize import brent_optimize

def optimization_result(title, fmin, xmin, K, f_ev, res=None):
    print(f"""
{title}
Optimization {res}
x minimum: {xmin},
f minimum: {fmin},
number of iterations: {K},
number of function evaluations: {f_ev},
""") if res == 'succes' else print(f"""{title}\nOptimization {res}""")

test_sqrt1 = [
    danilov,
    np.array([-2, 2]),
    0.001,
    'Square root func 1 test. Starting point (-2, 2)'
]

test_sqrt2 = [
    danilov,
    np.array([4, 3]),
    0.001,
    'Square root func 1 test. Starting point (4, 3)' 
]

test_rosen1 = [
    rosenbrok,
    np.array([-2, -1]),
    1e-4,
    'Rosenbrock1 test. Starting point (-2, -1)'
]

test_rosen2 = [
    rosenbrok,
    np.array([-3, 4]),
    1e-4,
    'Rosenbrock2 test. Starting point (-3, 4)'
]

test_rosen3 = [
    rosenbrok,
    np.array([3, 3]),
    1e-4,
    'Rosenbrock3 test. Starting point (3, 3)'
]


test_himmel1 = [
    himmelblau,
    np.array([0, -4]),
    1e-4,
    'Himmelblau1 test. Starting point (0, -4)'
]

test_himmel2 = [
    himmelblau,
    np.array([10, 21]),
    1e-4,
    'Himmelblau1 test. Starting point (10, 21)'
]

test_himmel3 = [
    himmelblau,
    np.array([-5, 17]),
    1e-4,
    'Himmelblau1 test. Starting point (-5, 17)'
]


def hooke_jeeves(f, x0, epsilon, title, s_v=np.array([1, 1]), gamma=1.5):
    anim = Animate3D(f, x0, title)
    dim = len(x0)
    base_vectors = np.empty([dim], dtype=object)
    X_K = np.empty([dim])
    X_K = x0
    for i in range(dim):
        other_list = np.empty([dim])
        for j in range(dim):
            if j != i:
                other_list[j] = 0
            else:
                other_list[j] = 1
        base_vectors[i] = other_list

    X_K_TILDA = 0
    f_plus = 0
    f_minus = 0
    counter = 0
    f_ev = 0
    PREV_X_K = 0
    while (norm(X_K_TILDA - X_K) > epsilon or (X_K_TILDA == X_K).all()):
        step_vector = 0
        X_K_TILDA = X_K
        for i in range(dim):
            f_start = f(X_K_TILDA)
            f_ev += 1
            step_vector = s_v * base_vectors[i]
            f_plus = f(X_K_TILDA + step_vector)
            f_minus = f(X_K_TILDA - step_vector)
            f_ev += 2
            if f_plus < f_start and f_plus <= f_minus:
                X_K_TILDA = X_K_TILDA + step_vector
                anim.add(X_K_TILDA)
            elif f_minus < f_start and f_minus < f_plus:
                X_K_TILDA = X_K_TILDA - step_vector
                anim.add(X_K_TILDA)
        counter += dim
        if (X_K_TILDA == X_K).all():
            s_v = s_v / gamma
            continue
        elif norm(X_K_TILDA - X_K) < epsilon:
            return f(X_K), X_K, counter, f_ev, anim, 'succes'
        else:
            res = brent_optimize(lambda alp: f(alp * (X_K_TILDA - X_K) + X_K), 0, 500, 1e-4)
            alpha = res[0]
            f_ev += res[1]
            X_K =  alpha*(X_K_TILDA - X_K) + X_K
            anim.add(X_K)
            if f(X_K_TILDA) < f(X_K):
                X_K = X_K_TILDA
                anim.add(X_K)
    return f(X_K), X_K, counter, f_ev, anim, 'succes'


fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_sqrt1, s_v=np.array([1, 1]))
optimization_result(test_sqrt1[3], fmin, xmin, K, f_ev, res=res)
# load_animation(anim, "Sqrt", "Hooke-Jeeves", test_num=1, duration=8000)

fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_sqrt2, s_v=np.array([1, 1]))
optimization_result(test_sqrt2[3], fmin, xmin, K, f_ev, res=res)
load_animation(anim, "Sqrt", "Hooke-Jeeves", test_num=2, duration=10000)

# fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_rosen1, s_v=np.array([1, 1]))
# optimization_result(test_rosen1[3], fmin, xmin, K, f_ev, res=res)
# load_animation(anim, "Rosenbrock", "Hooke-Jeeves", test_num=1, duration=10000)

# fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_rosen2, s_v=np.array([1, 1]))
# optimization_result(test_rosen2[3], fmin, xmin, K, f_ev, res=res)
# load_animation(anim, "Rosenbrock", "Hooke-Jeeves", test_num=2, duration=10000)

# fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_rosen3, s_v=np.array([1, 1]))
# optimization_result(test_rosen3[3], fmin, xmin, K, f_ev, res=res)
# load_animation(anim, "Rosenbrock", "Hooke-Jeeves", test_num=3, duration=10000)

fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_himmel1, s_v=np.array([1, 1]))
optimization_result(test_himmel1[3], fmin, xmin, K, f_ev, res=res)
load_animation(anim, "Himmelblau", "Hooke-Jeeves", test_num=1, duration=10000)

fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_himmel2, s_v=np.array([1, 1]))
optimization_result(test_himmel2[3], fmin, xmin, K, f_ev, res=res)
load_animation(anim, "Himmelblau", "Hooke-Jeeves", test_num=2, duration=10000)

fmin, xmin, K, f_ev, anim, res = hooke_jeeves(*test_himmel3, s_v=np.array([1, 1]))
optimization_result(test_himmel3[3], fmin, xmin, K, f_ev, res=res)
load_animation(anim, "Himmelblau", "Hooke-Jeeves", test_num=3, duration=10000)
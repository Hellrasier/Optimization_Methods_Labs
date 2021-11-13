import numpy as np
from numpy.linalg import norm
import matplotlib.animation as pltanimation
from test_functions import *
from animations import AnimateSimplex

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


from operator import itemgetter

class Simplex:
    def __init__(self, vertexes, f):
        self.f = f
        self.V = sorted(
            [(v, f(v)) for v in vertexes], 
            key=itemgetter(1)
        )
        self.n = len(vertexes) - 1
        
    def get_points(self):
        return [v[0] for v in self.V]
    
    def insert(self, xnew, f_new, i):
        self.V = [self.V[k] if k != i else (xnew, f_new) for k in range(len(self.V))]
    
    def min(self):
        return self.V[0]
    
    def max(self):
        return self.V[self.n]
    
    def max2(self):
        return self.V[self.n-1]
    
    def sort(self):
        self.V = sorted(self.V, key=itemgetter(1))
    
    def reflect(self, x, alpha):
        vs =  [v for v, f_v in self.V if (v != x).any()]
        new_x = np.divide(1 + alpha, self.n) * sum(vs) - alpha*x
        return new_x, self.f(new_x)
    
    def scale(self, x, beta):
        vs = [v for v, f_v in self.V if (v != x).any()]
        new_x = np.divide(1 - beta, self.n) * np.sum(vs) + beta*x
        return new_x, self.f(new_x)
    
    def reduction(self, delta):
        vertexes = [v for v, f_v in self.V]
        vertexes = [vertexes[0] + delta*(vertexes[i] - vertexes[0]) for i in range (1, self.n+1)]
        self.V = [self.V[0]] + [(v, self.f(v)) for v in vertexes]
    
    def stop_check(self, epsilon):
        for i in range(self.n+1):
            a, b = 0, 0
            if i == self.n:
                a, b = self.V[i][0], self.V[0][0]
            else:
                a, b = self.V[i][0], self.V[i+1][0]
            if norm(a - b) > epsilon:
                return False
        return True
    
    def print(self):
        print("Points:", *[v for v, f_v in self.V], "Func vals:", *[f_v for v, f_v in self.V])

def first_vs(base, l):
    n = len(base)
    return [base] + [base + l*np.eye(n)[i] for i in range(0, n)]
    
def nelder_mead(f, x0, epsilon, title, l=1, alpha=1, beta=2, gamma=0.5, delta=0.5):
    try:
        vs = first_vs(x0, l)
        smplx = Simplex(vs, f)
        anim = AnimateSimplex(f, vs, title)
    #     smplx.print()
        n = len(x0)
        k = 1
        f_ev = n + 1
        while not smplx.stop_check(epsilon):
            xm, fm = smplx.min()
            xh, fh = smplx.max() 
            xg, fg = smplx.max2()
            new_x, new_f = smplx.reflect(xh, alpha)
            f_ev += 1
            if new_f < fm:
                str_x, str_f = smplx.scale(new_x, beta)
                f_ev += 1
                if str_f < new_f:
                    smplx.insert(str_x, str_f, smplx.n)
                else:
                    smplx.insert(new_x, new_f, smplx.n)
            elif new_f < fg:
                smplx.insert(new_x, new_f, smplx.n)
            elif new_f < fh:
                smplx.insert(new_x, new_f, smplx.n)
                com_x, com_f = smplx.scale(new_x, gamma)
                f_ev += 1
                if com_f < new_f:
                    smplx.insert(com_x, com_f, smplx.n)
                else: 
                    smplx.reduction(delta)
                    f_ev += n
            else:
                com_x, com_f = smplx.scale(xh, gamma)
                if com_f < fh:
                    smplx.insert(com_x, com_f, smplx.n)
                else: 
                    smplx.reduction(delta)
                    f_ev += n

            smplx.sort()
            k += 1
            anim.add(smplx.get_points())
    #         smplx.print()
            if k == 30000:
                return *smplx.min(), k, f_ev, anim, 'fail'
        return *smplx.min(), k, f_ev, anim, 'succes'
    except:
        return *smplx.min(), k, f_ev, anim, 'fail'


# fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_sqrt1, l=1)
# optimization_result(test_sqrt1[3], fmin, xmin, K, f_ev, res=res)
# a = anim.get_animation(duration=10000).save('examples/Sqrt/Sqrt1-Nelder-Mead.gif')

# fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_sqrt2, l=1)
# optimization_result(test_sqrt2[3], fmin, xmin, K, f_ev, res=res)
# a = anim.get_animation(duration=10000).save('examples/Sqrt/Sqrt2-Nelder-Mead.gif')

fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_rosen1, l=1)
optimization_result(test_rosen1[3], fmin, xmin, K, f_ev, res=res)
a = anim.get_animation(duration=10000).save('examples/Rosenbrock/Rosenbrock1-Nelder-Mead.gif')

fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_rosen2, l=1)
optimization_result(test_rosen2[3], fmin, xmin, K, f_ev, res=res)
a = anim.get_animation(duration=10000).save('examples/Rosenbrock/Rosenbrock2-Nelder-Mead.gif')

fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_rosen3, l=0.6)
optimization_result(test_rosen3[3], fmin, xmin, K, f_ev, res=res)
a = anim.get_animation(duration=10000).save('examples/Rosenbrock/Rosenbrock3-Nelder-Mead.gif')

fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_himmel1, l=1)
optimization_result(test_himmel1[3], fmin, xmin, K, f_ev, res=res)
a = anim.get_animation(duration=10000).save('examples/Himmelblau/Himmel1-Nelder-Mead.gif')

fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_himmel2, l=1)
optimization_result(test_himmel2[3], fmin, xmin, K, f_ev, res=res)
a = anim.get_animation(duration=10000).save('examples/Himmelblau/Himmel2-Nelder-Mead.gif')

fmin, xmin, K, f_ev, anim, res = nelder_mead(*test_himmel3, l=1)
optimization_result(test_himmel3[3], fmin, xmin, K, f_ev, res=res)
a = anim.get_animation(duration=10000).save('examples/Himmelblau/Himmel3-Nelder-Mead.gif')

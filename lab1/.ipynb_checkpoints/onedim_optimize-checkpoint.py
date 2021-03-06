from scipy.misc import derivative
import numpy as np
import math as m
def der(f, x, n):
    return derivative(f, x, dx=1e-5, n=n)

def newton_method(f, x, epsilon):
    fd1 = der(f, x, 1)
    fd2 = der(f, x, 2)
    x2 = x - np.divide(fd1, fd2)
    k = 1
    while(m.fabs(x - x2) > epsilon):
        x = x2
        fd1 = der(f, x, 1)
        fd2 = der(f, x, 2)
        x2 = x - np.divide(fd1, fd2)
        k += 1
    return f(x2), x2, k

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
    return fm, xm, k+2

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
    F = getFibbonachies(epsilon)
    N = len(F) - 2
    l = b - a
    delta = epsilon / 100
    x2 = a + F[N]/F[N+1] * l + (-1)**(N+1)*2*delta/F[N+1]
    x1 = a + F[N-1]/F[N+1] * l
    f_x1, f_x2 = f(x1), f(x2)
    k = 1
    print(a, x1, x2, b)
    a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2)
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
        print(a, x1, x2, b)
        a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2) 
    print(a, b)
    return f_x, a, b, k+1

def one_dim_optimizer(f, a, b, epsilon):
    f_x, a, b, k1 = fibbonaci_method(f, a, b, epsilon * 1000)
    f_x, x, k2 = quadratic_approx(f, a, b, epsilon)
    return x, k1 + k2
    
from scipy.misc import derivative
import numpy as np
from genetical_algorithms import utils, operators
import math as m
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
from scipy.constants import golden
def der(f, x, n):
    return derivative(f, x, dx=np.float64(1e-7), n=n)

# def der2(f, x):
#     eps = np.float64(1e-6)
#     der1 = der(f, x, 1)
#     x_dx = x + eps
#     der2 = der(f, x_dx, 1)
#     return np.divide(der2 - der1, eps)

def brent_optimize(f, a, b, epsilon):
    res = minimize_scalar(f)
    return np.float64(res.x), res.nfev
    
def getFibbonachies(n):
    fibbs = [] 
    fibbs.append(1)
    fibbs.append(1)
    while(fibbs[len(fibbs) - 1] < n):
        f2 = fibbs[len(fibbs) - 1]
        f1 = fibbs[len(fibbs) - 2]
        fibbs.append(f1 + f2)
    return fibbs
    
def fibbonaci_method(f, a, b, epsilon, iters=None):
    if iters != None:
        F = getFibbonachies(iters)
    else:
        F = getFibbonachies(1/epsilon)
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
    f_ev = 0
    x0 = a
    der1 = der(f, x0, 1)
    sec_der = der(f, x0, 2)
    while sec_der <= epsilon:
        print(x0)
        x1, k = fibbonaci_method(f, a, b, epsilon, iters=10)
        f_ev += k
        if m.fabs(x0 - x1) > epsilon:
            return x1, f_ev
        sec_der = der(f, x0, 2)
        x0 = x1
    x1 = x0 - np.divide(der1, sec_der)
    print(x1)
    der2 = der(f, x1, 1)
    f_ev += 2
    while(der2 > epsilon):
        while np.absolute(der2 - der1) < epsilon:
            x1, k = fibbonaci_method(f, x1, b, epsilon, iters=10)
            f_ev += k
            if np.absolute(der2) > epsilon:
                return x1, f_ev
            der1 = der2
            der2 = der(f, x1, 1)
        x0 = x1
        x2 = x1 - np.divide(x1 - x0, der2 - der1) * der2
        x0 = x1
        x1 = x2
        der1 = der2
        der2 = der(f, x1, 1)
        f_ev += 2
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
    f_ev = 0
    x0 = a
    der1 = der(f, x0, 1)
    sec_der = der(f, x0, 2)
    if sec_der < 0:
        return brent_optimize(f, a, b, eps)
    x1 = x0 - np.divide(der1, sec_der)
    der2 = der(f, x1, 1)
    f_ev += 2
    while(np.absolute(x0 - x1) > epsilon):
        x0 = x1
        if (der2 - der1 == 0):
            return brent_optimize(f, a, b, eps)
        x2 = x1 - np.divide(x1 - x0, der2 - der1) * der2
        x0 = x1
        x1 = x2
        der1 = der2
        der2 = der(f, x1, 1)
        f_ev += 2
    return x1, f_ev   

def newton_modified2(f, a, b, epsilon):
    f_ev = 0
    x0 = a
    print(x0, f(x0))
    der1 = der(f, x0, 1)
    sec_der = der(f, x0, 2)
    f_ev += 4
    if sec_der < 0:
        return fibbonaci_method(f, a, b, epsilon)
    x1 = x0 - np.divide(der1, sec_der)
    print(x1, f(x1))
    der2 = der(f, x1, 1)
    f_ev += 2
    while(np.absolute(x0 - x1) > epsilon):
        x0 = x1
        if (der2 - der1 == 0):
            return fibbonaci_method(f, x0, b, epsilon)
        x2 = x1 - np.divide(x1 - x0, der2 - der1) * der2
        print(x2, f(x2))
        x0 = x1
        x1 = x2
        der1 = der2
        der2 = der(f, x1, 1)
        f_ev += 2
    print(x1, f(x1))
    return x1, f_ev   

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
    while(x2 - x1 > epsilon):
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


def golden_scale_method(f, a, b, epsilon, iterations=None):
    g_c = golden
    x, f_x = 0.0, 0.0
    x2 = (b - a)/g_c + a 
    x1 = b - (b - a)/(g_c)
    f_x1, f_x2 = f(x1), f(x2)
    f_ev = 2
    k = 1
    a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2)
    while(b - a > epsilon):
        y = a + b - x
        if x < y:
            x1, x2 = x, y
            f_x1, f_x2 = f_x, f(y)
            f_ev += 1
        else: 
            x1, x2 = y, x
            f_x1, f_x2 = f(y), f_x
            f_ev += 1
        a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2) 
        k += 1 
        if iterations != None:
            if k == iterations:
                return a, b, f_ev
    return a, b, f_ev


def qubic_modified(f, a, b, epsilon):
    print(a, b)
    x1, x2, f_ev = golden_scale_method(f, a, b, epsilon, iterations=3)
    print(x1, x2)
    f1, f2 = f(x1), f(x2)
    df1, df2 = der(f, x1, 1), der(f, x2, 1)
    f_ev += 6
    while(df1*df2 > 0):
        print("starting golden")
        x1, x2, k = golden_scale_method(f, x1, x2, epsilon, iterations=5)
        print(x1, x2)
        f1, f2 = f(x1), f(x2)
        f_ev += k
        df1, df2 = der(f, x1, 1), der(f, x2, 1)
        if x2 - x1 < epsilon:
            return (x1 + x2)/2, f_ev
    xm = (x1 + x2)/2
    while(True):
        mu = find_mu(x1, x2, f1, f2, df1, df2)
        xm = x1 + mu*(x2 - x1)
        dfm = der(f, xm, 1)
        fm = f(xm)
        f_ev += 3
        if xm > x2 or (fm > f1 and fm > f2) or df1*df2 > 0:
            print("starting golden")
            x1, x2, k = golden_scale_method(f, x1, x2, epsilon, iterations=5)
            print(x1, x2)
            f1, f2 = f(x1), f(x2)
            f_ev += k
            df1, df2 = der(f, x1, 1), der(f, x2, 1)
            if x2 - x1 < epsilon:
                return (x1 + x2)/2, f_ev
            continue
        print(x1, xm, x2, f(x1), f(xm), f(x2))

        if x2 - xm < epsilon or xm - x1 < epsilon:
            return xm, f_ev
        if df1*dfm < 0:
            x2 = xm
            f2 = fm
            df2 = dfm
        else:
            x1 = xm
            f1 = fm
            df1 = dfm
    

# def golden_iteration(a, b, x, f_x, f_ev):
#     y = a + b - x
#     if x < y:
#         x1, x2 = x, y
#         f_x1, f_x2 = f_x, f(y)
#         k += 1
#     else: 
#         x1, x2 = y, x
#         f_x1, f_x2 = f(y), f_x
#         k += 1
#     a, b, x, f_x = cut_section_procedure(a, x1, x2, b, f_x1, f_x2) 
#     f_ev += 1
#     return a, b, x, f_x, f_ev

def brent_root_algorithm(f, a, b):
    df = lambda x: der(f, x, 1)
    res = root_scalar(df, bracket=[a, b], method="brenth")
    print("root", res.root)
    return np.float64(res.root), res.function_calls

def one_dim_optimizer(f, a, b, eps):
    x, f_ev = 0, 0
    if der(f, a, 1)*der(f, b, 1) < 0:
        return newton_modified(f, a, b, eps)
    else:
        return brent_optimize(f, a, b, eps)

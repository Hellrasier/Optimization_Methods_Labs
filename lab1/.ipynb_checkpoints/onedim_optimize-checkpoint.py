from scipy.misc import derivative
import math as m
def der(f, x, n):
    return derivative(f, x, dx=1e-5, n=n)

def newton_method(f, x, epsilon):
    fd1 = der(f, x, 1)
    fd2 = der(f, x, 2)
    x2 = x - fd1/fd2
    k = 1
    while(m.fabs(x - x2) > epsilon):
        x = x2
        fd1 = der(f, x, 1)
        fd2 = der(f, x, 2)
        x2 = x - fd1/fd2
        k += 1
    return f(x2), x2, k
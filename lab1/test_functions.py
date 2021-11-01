import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from itertools import chain

def plot_2Dfunction(f, zlim=[-10, 10], xy_range=[-10, 10]):
    X = np.arange(*xy_range, 0.01)
    Y = np.arange(*xy_range, 0.01)
    X, Y = np.meshgrid(X, Y)
    get_f = lambda i: (f(np.array([X[i, j], Y[i, j]])) for j in range(X.shape[1]))
    a = np.fromiter(chain.from_iterable(get_f(i) for i in range(X.shape[0])), float, X.shape[0]*X.shape[1])
    Z = a.reshape(X.shape)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=1000, ccount=1000,
                           linewidth=0, antialiased=False)
    ax.set_zlim(*zlim)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Rastrigin function
# global minimum: f(0, ..., 0) = 0
#  search domain: -5.12 <= x_i <= 5.12
rastrigin = lambda x: 10 * x.shape[0] + np.sum(np.power(x, 2) - 10 * np.cos(2*np.pi*x))

# Ackley function
# global minimum: f(0, 0) = 0
#  search domain: -5 <= x, y <= 5
ackley = lambda x: (-20 * np.exp(-0.2 * np.sqrt(0.5 * (np.power(x[0], 2) + np.power(x[1], 2)))) -
    np.exp(0.5 * (np.cos(2*np.pi*np.power(x[0], 2)) + np.cos(2 * np.pi * np.power(x[1], 2))) ) + np.e + 20)

# Sphere function
# gloabl minimum: f(0, ..., 0) = 0
sphere = lambda x: np.sum(np.power(x, 2))

# Rozenbrock
# f(1, 1) = 0
rozenbrock = lambda x: 100*np.power(x[1]-x[0]**2, 2) + np.power(1-x[0], 2)

# f(3, 0.5) = 0
beale = lambda x: np.power(1.5 - x[0] + x[0]*x[1], 2) + np.power(2.25 - x[0] + x[0]*x[1]**2, 2) + np.power(2.625 - x[0] + x[0]*x[1]**3, 2)

# f(0, -1) = 3
goldstein_price = lambda x: (
    (1 + np.power(x[0] + x[1] + 1, 2) * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) *
    (30 + np.power(2*x[0] - 3*x[1], 2) * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
)

if __name__ == "__main__":
    plot_2Dfunction(rozenbrock, zlim=[0, 300], xy_range=[0, 1.5])
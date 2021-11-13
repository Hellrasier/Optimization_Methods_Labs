from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as pltanimation
from itertools import chain
from matplotlib.ticker import LinearLocator


def plot_function(f, fig, ax, xy_range=[-10, 10], cmap=cm.gist_ncar, quality=100, bar=True):
    X = np.linspace(*xy_range, 1000)
    Y = np.linspace(*xy_range, 1000)
    X, Y = np.meshgrid(X, Y)
    get_f = lambda i: (f(np.array([X[i, j], Y[i, j]])) for j in range(X.shape[1]))
    a = np.fromiter(chain.from_iterable(get_f(i) for i in range(X.shape[0])), float, X.shape[0]*X.shape[1])
    Z = a.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, rcount=quality, ccount=quality,
    linewidth=0, antialiased=False, alpha=0.6)
    zlim = [Z.min(), Z.max()]
    ax.set_zlim(*zlim) 
    ax.zaxis.set_major_locator(LinearLocator(10)) 
    ax.zaxis.set_major_formatter('{x:.02f}')
    if bar:
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    return fig

def plot_contour(f, fig, ax, xy_range=[-10, 10], cmap=cm.gist_ncar, quality=100, bar=True):
    X = np.linspace(*xy_range, 1000)
    Y = np.linspace(*xy_range, 1000)
    X, Y = np.meshgrid(X, Y)
    get_f = lambda i: (f(np.array([X[i, j], Y[i, j]])) for j in range(X.shape[1]))
    a = np.fromiter(chain.from_iterable(get_f(i) for i in range(X.shape[0])), float, X.shape[0]*X.shape[1])
    Z = a.reshape(X.shape)
    plot = ax.contourf(X, Y, Z, np.linspace(Z.min(), Z.max(), 1000), cmap=cmap)
    if bar:
        fig.colorbar(plot, ax=ax, shrink=0.5, aspect=5)
    return fig

class Animate3D:
    anms = None 
    fig = plt.figure(figsize=(13, 20))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax2 = fig.add_subplot(2, 1, 2)
    counter = 0
    
    def __init__(self, f, x0, title):
        self.f = f
        self.data = np.array([x0])
        self.fig.suptitle(title, fontsize=16)
        self.plt_3d = lambda xy_range: plot_function(f, fig=self.fig, ax=self.ax1, xy_range=xy_range, cmap=cm.turbo, bar=False)
        self.contour = lambda xy_range: plot_contour(f, fig=self.fig, ax=self.ax2, xy_range=xy_range, cmap=cm.turbo, bar=False)
    
    def length(self):
        return len(self.data)
    
    def add(self, x):
        self.data = np.vstack([self.data, [x]])
            
    def data_min_max(self):
        mn = np.array(self.data).min()
        mx = max([x[0] for x in self.data] + [x[1] for x in self.data])
        return [mn - 1, mx + 1]
    
    def frame(self, i):
        if self.counter == 0:
            self.ax1.cla()
            self.ax2.cla()
            self.counter += 1
            return
        if(i % self.length() == 0):
            self.ax1.clear()
            self.ax2.clear()
            self.plt_3d(xy_range=self.data_min_max())
            self.contour(xy_range=self.data_min_max())
            x0 = self.data[0]
            self.point1 = self.ax1.scatter(x0[0], x0[1], self.f(x0), color='r',  s=6, marker='^')
            self.point2 = self.ax2.scatter(x0[0], x0[1], color='r', marker='s', s=3)
            self.ax1.view_init(azim=(45 + 90 * np.floor(i / self.length())))
            self.counter += 1
            return
        x1, x2 = self.data[(i-1) % self.length()], self.data[i % self.length()]
        X = np.linspace(x1[0], x2[0], 200)
        Y = np.linspace(x1[1], x2[1], 200)
        Z = [self.f(np.array([x, y])) for x,y in zip(X, Y)]
        self.point1 = self.ax1.scatter(x2[0], x2[1], self.f(x2), color='r', s=8, marker='^')
        self.point2 = self.ax2.scatter(x2[0], x2[1], color='r', s=3, marker='s')
        self.line1 = self.ax1.plot(X, Y, Z, color='r', linewidth=5)
        self.line2 = self.ax2.plot([x1[0], x2[0]], [x1[1], x2[1]], color='r', linewidth=0.7)
        self.ax2.set_xlabel(f"Iteration:{i % self.length()}")
        if i == self.length()*4 - 1:
            print("Animation created succesfully")
    
    def get_animation(self, duration):
        return pltanimation.FuncAnimation(self.fig, self.frame, frames=self.length()*4, interval=duration/self.length(), repeat=True)
    
    def destruct(self):
        plt.close('all')
    
def draw_simplex(points, fig, ax):
    [x1, x2, x3] = points
    point1 = ax.scatter(x1[0], x1[1], color='r', s=3, marker='s')
    point2 = ax.scatter(x2[0], x2[1], color='r', s=3, marker='s')
    point3 = ax.scatter(x3[0], x3[1], color='r', s=3, marker='s')
    line1 = ax.plot([x1[0], x2[0]], [x1[1], x2[1]], color='r', linewidth=0.7)
    line2 = ax.plot([x2[0], x3[0]], [x2[1], x3[1]], color='r', linewidth=0.7)
    line3 = ax.plot([x3[0], x1[0]], [x3[1], x1[1]], color='r', linewidth=0.7)
    return fig, ax

class AnimateSimplex:
    anms = None 
    fig, ax = plt.subplots(dpi=250)
    counter = 0
    
    def __init__(self, f, points, title):
        self.f = f
        self.data = np.array([points])
        self.fig.suptitle(title, fontsize=16)
        self.contour = lambda xy_range: plot_contour(f, fig=self.fig, ax=self.ax, xy_range=xy_range, cmap=cm.turbo)
#         self.simplex = draw_simplex(points, self.fig, self.ax)
        
    def length(self):
        return len(self.data)
    
    def add(self, points):
        self.data = np.vstack([self.data, [points]])
        
    def data_min_max(self):
         mn = np.array(self.data).min()
         mx = np.array(self.data).max()
         return [mn - 1, mx + 1]
        
    def frame(self, i):
        if self.counter == 0:
            self.ax.cla()
            self.counter += 1
            return
        if self.counter == 1:
            self.contour(xy_range=self.data_min_max())
            self.simplex = draw_simplex(self.data[0], self.fig, self.ax)
            self.counter += 1
        points = self.data[i]
        self.ax.set_xlabel(f"Iteration:{i}")
        self.fig, self.ax = draw_simplex(points, self.fig, self.ax)
        print(f"\rIteration: {i}/{len(self.data)}")
        if i == self.length() - 1:
            print("Animation created succesfully")
    
    def get_animation(self, duration):
        return pltanimation.FuncAnimation(self.fig, self.frame, frames=self.length(), interval=duration/self.length(), repeat=True)
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import  Axes3D
from scipy.optimize import minimize_scalar
import scipy as sp

from onedim_optimize import fibbonaci_method, upgraded_newton


# def ciclic_coordinate_descent(function,domain, epsilon, use_once, X_K_OPTIONAL):
#     number_of_base_vectors = len(domain)
#     base_vectors = np.empty([number_of_base_vectors], dtype=object)
#     X_K = np.empty([number_of_base_vectors])
#     X_K = X_K_OPTIONAL
#     if not(X_K_OPTIONAL):
#        X_K[0], X_K[1] = np.random.choice(domain[0],1),  np.random.choice(domain[1],1)
#
#     # print(function)
#     for i in range(number_of_base_vectors):
#         other_list = np.empty([number_of_base_vectors])
#         for j in range(number_of_base_vectors):
#             if j != i:
#                 other_list[j] = 0
#             else:
#                 other_list[j] = 1
#         base_vectors[i] = other_list
#     counter = 0
#     X_K_PREV = 0
#
#     while sp.linalg.norm(X_K - X_K_PREV) > epsilon:
#           counter +=1
#           phi = np.zeros(len(domain))
#           for i in range(number_of_base_vectors):
#               calculated_vector = np.zeros(len(domain))
#               for j in range(i):
#                   calculated_vector += base_vectors[j] * phi[j]
#               phi[i] = minimize_scalar(lambda alpha: function(x = X_K + alpha * base_vectors[i] + calculated_vector)).x
#           X_K_PREV = X_K
#           for i in range(number_of_base_vectors):
#               X_K = X_K + base_vectors[i] * phi
#           if use_once == True:
#               break;
#           # print(X_K)
#           # print(function(X_K))
#     return function(X_K), X_K;


# print(X);
# Z = ackley((X, Y))
# fig = plt.figure()
# ax = plt.axes(projection='2d')
# ax.plot_surface(X,Y, rstride=1,cstride=1,cmap='jet',edgecolor='none')
# plt.show()

def rozenbrock(x):
    return 100*np.power(x[1]-x[0]**2, 2) + np.power(1-x[0], 2)

atetkov  = lambda x: 6*pow(x[0],2) - 4*x[0]*x[1] + 3*pow(x[1],2) + 4* math.sqrt(5)*(x[0] + 2*x[1]) + 22


# R_n = [np.linspace(-10,10,np.power(10,4)),np.linspace(-10,10,np.power(10,4))]
# print(R_n)
# X_K[0], X_K[1] = np.random.choice(domain[0], 1), np.random.choice(domain[1], 1)

def  Hooke_Jeeves(f,x0, start_vector,epsilon, gamma=1.5):

     dim = len(x0)
     base_vectors = np.empty([dim], dtype=object)
     X_K = np.empty([dim])
     X_K = x0
     # X_K[0], X_K[1] = -2,1
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
     counter_func = 0
     PREV_X_K = 0

     while (sp.linalg.norm(X_K_TILDA - X_K) > epsilon or  (X_K_TILDA == X_K).all()) :

           counter += 1
           step_vector = 0
           X_K_TILDA = X_K

           for i in range(dim):
               f_start = f(X_K_TILDA)
               step_vector = start_vector * base_vectors[i]
               f_plus = f(X_K_TILDA + step_vector)
               f_minus = f(X_K_TILDA - step_vector)

               if f_plus < f_start and f_plus <= f_minus:
                   X_K_TILDA = X_K_TILDA + step_vector
               elif f_minus < f_start and f_minus < f_plus:
                   X_K_TILDA = X_K_TILDA - step_vector

           if (X_K_TILDA == X_K).all():
              start_vector = start_vector / gamma
              continue
           elif sp.linalg.norm(X_K_TILDA - X_K) < epsilon:
               return f(X_K), X_K,counter
           else:
               # res = minimize_scalar(lambda alp: f(alp*(X_K_TILDA - X_K) + X_K))
               # alpha = fibbonaci_method(lambda alp: f(alp*(X_K_TILDA - X_K) + X_K), 0, 500, 1e-6)
               # res = fibbonaci_method(lambda alp: f(alp * (X_K_TILDA - X_K) + X_K), 0, 500, 1e-6)
               res = upgraded_newton(lambda alp: f(alp * (X_K_TILDA - X_K) + X_K), 0, 500, 1e-4)
               alpha = 0

               alpha = res[0]
               counter_func += res[1]
               X_K =  alpha*(X_K_TILDA - X_K) + X_K
               if f(X_K_TILDA) < f(X_K):
                  X_K = X_K_TILDA

     return f(X_K), X_K, counter, counter_func

# f(3, 0.5) = 0
beale = lambda x: np.power(1.5 - x[0] + x[0]*x[1], 2) + np.power(2.25 - x[0] + x[0]*x[1]**2, 2) + np.power(2.625 - x[0] + x[0]*x[1]**3, 2)
sphere = lambda x: np.sum(np.power(x, 2))
# f(0, -1) = 3
goldstein_price = lambda x: (
    (1 + np.power(x[0] + x[1] + 1, 2) * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)) *
    (30 + np.power(2*x[0] - 3*x[1], 2) * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
)


# f( 3,         2)        = 0
# f(-2.805118,  3.131312) = 0
# f(-3.779310, -3.283186) = 0
# f( 3.584428, -1.848126) = 0
himmelblau = lambda x: np.power(x[0]**2 + x[1] - 11, 2) + np.power(x[0] + x[1]**2 - 7, 2)


# f(-10, 1) = 0
bukin = lambda x: 100 * np.sqrt(np.abs(x[1] - 0.01*x[0]**2)) + 0.01*np.abs(x[0]+10)

k = Hooke_Jeeves(rozenbrock,np.array([3,2]),np.array([4,4]),1e-6)
print(k)


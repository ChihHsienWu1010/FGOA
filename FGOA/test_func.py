# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:24:06 2023

@author: user
"""
import numpy as np

#test function
def three_hump_camel(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6 / 6 + x[0]*x[1] + x[1]**2

def sphere(x):
    return np.sum(np.square(x))

def sum_squares(x):
    return sum((i + 1) * x_i ** 2 for i, x_i in enumerate(x))

def beale(x):
    x1, x2 = x[0], x[1]
    return (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2

def colville(x):
    return 100 * (x[0]**2 - x[1])**2 + (1 - x[0])**2 + 90 * (x[1]**2 - x[2])**2 + (1 - x[1])**2 + 10.1 * ((x[2] - 1)**2 + (x[3] - 1)**2) + 19.8 * (x[2] - 1) * (x[3] - 1)

def zakharov(x):
    return sum(x_i ** 2 for x_i in x) + (sum(0.5 * i * x_i for i, x_i in enumerate(x))) ** 2 + (sum(0.5 * i * x_i for i, x_i in enumerate(x))) ** 4

def schwefel(x):
    n = len(x)
    return 418.9829 * n - sum(x_i * np.sin(np.sqrt(abs(x_i))) for x_i in x)

def dixon_price(x):
    n = len(x)
    sum_term = sum((i + 1) * (2 * x[i]**2 - x[i - 1])**2 for i in range(1, n))
    return (x[0] - 1)**2 + sum_term

def holder_table(x):
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2)/np.pi)))

def matyas(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

def levy(x):
    return np.sin(3 * np.pi * x[0])**2 + (x[0] - 1)**2 * (1 + np.sin(3 * np.pi * x[1])**2) + (x[1] - 1)**2 * (1 + np.sin(2 * np.pi * x[1])**2)

def levy_n13_function(x):
    d = len(x)
    w = [1 + (xi - 1) / 4 for xi in x]
    term1 = (np.sin(np.pi * w[0]))**2
    term2 = sum([(w[i] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2) for i in range(d-1)])
    term3 = (w[d-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[d-1]))**2)
    return term1 + term2 + term3

def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum(x**2 - A * np.cos(2 * np.pi * x))

def schaffer_n4_function(x):
    term1 = np.sin((x[0]**2 + x[1]**2)**0.5)
    term2 = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + (term1**2 - 0.5) / term2

def schaffer_n2_function(x):
    term1 = (np.sin(x[0]**2 - x[1]**2))**2 - 0.5
    term2 = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + term1 / term2

def drop_wave_function(x):
    numerator = 1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))
    denominator = 0.5 * (x[0]**2 + x[1]**2) + 2
    return -numerator / denominator

def rosenbrock(x):
    # return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    a = 1
    b = 100
    return sum((a - x[:-1])**2 + b*(x[1:] - x[:-1]**2)**2)

def griewank_function(x):
    d = len(x)
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(np.array(x) / np.sqrt(np.arange(1, d+1))))
    return term1 - term2 + 1

def ackley(x):
    d = len(x)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

#CEC2017
def bent_cigar(x):
    return x[0]**2 + 10**6 * np.sum(x[1:]**2)

def shifted_sphere(x):
    o=30
    bias=40
    z = x - o
    return np.sum(z**2) + bias

def sum_diff_pow(x: np.ndarray) -> float:
    i = np.arange(1, len(x) + 1)
    x_pow = np.abs(x) ** i
    return np.sum(x_pow)

def shifted_schwefel_1_2(x):
    o=30
    bias=40
    z = x - o
    D = len(z)
    result = sum([(sum(z[:i+1]))**2 for i in range(D)])
    return result + bias

def shifted_rosenbrock_function(x):
    shift_vector=30
    shifted_x = x - shift_vector
    n = len(shifted_x)
    return sum(100.0 * (shifted_x[i] - shifted_x[i-1]**2.0)**2.0 + (1 - shifted_x[i-1])**2.0 for i in range(1, n))

def shifted_rotated_rastrigin_function(x):
    shift_vector=30
    rotation_matrix=30
    shifted_x = x - shift_vector
    rotated_x = np.dot(rotation_matrix, shifted_x)
    n = len(rotated_x)
    return 10 * n + sum(rotated_x[i]**2 - 10 * np.cos(2 * np.pi * rotated_x[i]) for i in range(n))

def katsuura(x):
    n = len(x)
    product_term = 1
    for i in range(n):
        sum_term = 0
        for j in range(1, 33):
            sum_term += abs(2**j * x[i] - round(2**j * x[i])) * 2**j
        product_term *= (1 + (i+1) * sum_term) ** (10/n**1.2)
    return (10/n**2) * product_term - (10/n**2)

def high_conditioned_elliptic(x):
    factor = 6
    i = np.arange(len(x))
    sm = x**2 * 10**(i * factor)
    return np.sum(sm)

def weierstrass(x):
    xy_values = 0.005 * x
    k = np.arange(0, 21)
    ak = 0.5**k
    bk = np.pi * (3**k)
    
    kcs = ak * np.cos(2 * (xy_values[:, np.newaxis] + 0.5) * bk)
    ksm = np.sum(kcs, axis=1)
    sm = np.sum(ksm)
    
    kcs = ak * np.cos(bk)
    ksm_const = np.sum(kcs)
    return sm - len(xy_values) * ksm_const

def discus(x):
    sm0 = 1e+6 * x[0]**2
    sm = np.sum(x[1:]**2)
    return sm0 + sm
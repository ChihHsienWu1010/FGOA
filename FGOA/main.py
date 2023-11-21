# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:21:49 2023

@author: user
"""
import numpy as np
from fgoa import FGOA
from test_func import schwefel

# Test the adjusted algorithm on a high-dimensional problem
dim = 2
size = 50
minx = -500
maxx = 500
iter = 1000
incentive_threshold = 0.8
fatigue = 500  # Increased fatigue threshold
o = np.random.uniform(low=minx, high=maxx, size=dim) # Random shift for testing

fgoa = FGOA(dim, size, minx, maxx, iter, incentive_threshold, fatigue)
# adjusted_gbest_ackley, adjusted_gbest_score_ackley = adjusted_pso_ackley.optimize(lambda x: shifted_schwefel_1_2(x, o))
adjusted_gbest_ackley, adjusted_gbest_score_ackley = fgoa.optimize(schwefel)

print("Optimal solution for function:", adjusted_gbest_ackley)
print("Optimal objective for function IPSO:", adjusted_gbest_score_ackley) 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 04-03-21
File: ej_01.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt

def BHEEcExacta(n0, t, r, K):
    num = K * n0
    den = n0 + (K - n0) * np.power(r,t)
    return num/den

def BevertonHolt(n_t, r, K):
    return (r*n_t)/(1+((r-1)/K)*n_t)

# CI = [0,0.4,0.8,1.2,1.6,2,2.4,2.8]
# CI = np.arange(-1,5,0.2)
# CI = np.linspace(-4,3,22)
CI = np.linspace(-3,3,11)
r = 0.5
K = 2
iteraciones = 15
t = np.array(list(range(iteraciones+1)))

for n_0 in CI:
    n_t = np.array([n_0])

    for i in range(iteraciones):

        n_next = BevertonHolt(n_t[-1], 0.5, 2)
        n_t = np.append(n_t, n_next)

    plt.plot(t, n_t, label=n_0)

for n_0 in CI:
    



# n_t = np.array([0.5])


# for i in range(100):
#     n_next = BevertonHolt(n_t[-1], 0.5, 2)

#     n_t = np.append(n_t, n_next)

# plt.plot(t, n_t)
# plt.legend(loc='best')
plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 09-03-21
File: ej_04.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt


s = np.random.poisson(5, 100000)

# count, bins, ignored = plt.hist(s, 14, density=True)
# plt.show()

def ej4(Nt=20):

    N = np.array([1.0])

    for _ in range(1,Nt):
        r = np.random.poisson(1.7)
        n = r * N[-1]               # Calculamos la siguiente iteracion
        N = np.append(N,n)
    
    return N

plt.plot(ej4(), 'bo')
plt.show()



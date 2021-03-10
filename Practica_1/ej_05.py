#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 10-03-21
File: ej_05.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def logisticModel(N,t,r,K):
    dNdt = r * N * (1 - N/K)
    return dNdt

def getDisasterTime(p,l):
    return -l*np.log(1-p)

# pp = np.linspace(0,1-1e-8,1000)

# plt.plot(pp, getDisasterTime(pp, 0.5))
# plt.show()

# El valor de K no importa mucho, es el valor al cual va a converger
K = 10

t_final = 100
N0 = K/2


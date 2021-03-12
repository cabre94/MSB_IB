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

def getDisasterTime(q,l):
    return -np.log(1-q) / l

"""
Calculo auxiliar
"""
# n0_kk = np.linspace(0,15,16)
# t_kk = np.linspace(0,10,1000)
# for i in n0_kk:
#     n_kk = odeint(logisticModel, i, t_kk, args=(1,10.0))
#     plt.plot(t_kk, n_kk)
# plt.show()

# pp = np.linspace(0,1-1e-8,1000)

# plt.plot(pp, getDisasterTime(pp, 0.5))
# plt.show()

# El valor de K no importa mucho, es el valor al cual va a converger
K = 10

t_final = 500
N0 = K/2
# N0 = K*2

r = 1     # Mas grande -> + rapido evolucina -> + chance de sobrevivir 
p = 0.5     # Fraccion que sobrevive
l = 1.4       # 

N = np.array([])
t_log = np.array([])

t_current = 0

while(t_current < t_final):

    # Tomamos un numero aleatorio entre 0 y 1, con el que vamos a determinar
    # el tiempo del siguiente desastre
    q = np.random.random()

    # Obtenemos el tiempo del siguiente desastre
    t_disaster = getDisasterTime(q,l)

    # Integro entre desastres
    t = np.linspace(t_current, t_current+t_disaster, 1000)
    n = odeint(logisticModel, N0, t , args=(r,K))

    # Agregamos la evolucion a los datos guardados
    N = np.append(N, n)
    t_log = np.append(t_log, t)

    # Aplicamos el efecto del desastre, cambiando las CI de la prox iteracion
    N0 = N[-1]*p
    t_current += t_disaster

plt.plot(t_log, N)
plt.show()

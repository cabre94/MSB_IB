#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 04-03-21
File: ej_02.py
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
import seaborn as sns
sns.set()

SAVE_PATH = "Figuras/Ej_02"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Ecuacion diferencial del modelo
def model(N, t, r, K, N_T):
    dNdt = r * N * (1 - N_T / K)
    return dNdt

# Solucion aproximada
def aproximacion(t, epsi, C=1):

    pT = C * np.exp((epsi * t) / (1 + np.pi * np.pi * 0.25))
    sT = np.cos(t * (1 - ((epsi * np.pi) / (2 * (1 + np.pi * np.pi * 0.25)))))

    return 1 + pT * sT

# Funcion que resuelve la ec. diferencial para un r, T y K dados
def solve(r, T, K, ite=10001, Nlim=50, N0=2):

    t = np.linspace(0,Nlim,ite)
    N = np.zeros_like(t)
    N_delay = np.zeros_like(t)
    N_delay[t<T] = N0
    N_delay = N_delay[N_delay != 0]

    N[0] = N0

    for i in range(1,ite):
        tspan = [t[i-1], t[i]]

        n = odeint(model, N0, tspan, args=(r,K, N_delay[i]))

        N[i] = n[1]
        N_delay = np.append(N_delay, n[1])
        N0 = n[1]
    
    return [N, N_delay, t]

def a():
    kk1 = solve(r=0.3, T=1, K=10)
    kk2 = solve(r=1.2, T=1, K=10)
    kk3 = solve(r=2.0, T=1, K=10)
    plt.figure()
    plt.plot(kk1[2], kk1[0])
    plt.plot(kk2[2], kk2[0])
    plt.plot(kk3[2], kk3[0])
    # plt.show()

"""
Segunda parte del ejercicio
"""





epsi = 1 - np.pi / (2*2)
plt.figure()
plt.plot(t, N, label='Numerico')
plt.plot(t, aproximacion(t, epsi), label=r'$\varepsilon={:.2f}$'.format(epsi))
plt.plot(t, aproximacion(t, 1e-5), label=r'$\varepsilon=10^{-5}$')
plt.legend(loc='best')
plt.tight_layout()
# os.path.join(SAVE_PATH, "Coweb_r={}".format(r))
# plt.savefig()
plt.show()


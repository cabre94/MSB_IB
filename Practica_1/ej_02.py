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

    # Primer termino del choclo
    pT = C * np.exp((epsi * t) / (1 + np.pi * np.pi * 0.25))
    # Segundo termino del choclo
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

# Funcion para resolver la primera parte del ejercicio, usando distintos
# parametros para ver los distintos z
def a():
    kk1 = solve(r=0.3, T=1, K=10)
    kk2 = solve(r=1.2, T=1, K=10)
    kk3 = solve(r=2.0, T=1, K=10)
    
    plt.figure()
    plt.plot(kk1[2], kk1[0], label="r=0.3")
    plt.plot(kk2[2], kk2[0], label="r=1.2")
    plt.plot(kk3[2], kk3[0], label="r=2.0")
    plt.xlabel("t")
    plt.legend(loc='best')
    plt.tight_layout()
    file_name = os.path.join(SAVE_PATH, "Regimenes")
    plt.savefig(file_name, format='pdf')
    plt.show()

"""
Segunda parte del ejercicio
"""



# kk = solve(2.0,1,10)

r = 2.0
T_c = np.pi / (2.0*2.0)
# epsi = 1e-5
# T = T_c + epsi
T = 1
epsi = T - T_c

kk = solve(r,T,K=10,Nlim=100)

plt.figure()
plt.plot(kk[2], kk[0], label='Numerico')
plt.plot(kk[2], aproximacion(kk[2], epsi), label=r'$\varepsilon={:.2f}$'.format(epsi))
# plt.plot(kk[2], aproximacion(kk[2], 1e-5), label=r'$\varepsilon=10^{-5}$')
plt.legend(loc='best')
plt.tight_layout()
file_name = os.path.join(SAVE_PATH, "Comp_Analitica")
plt.savefig(file_name, format='pdf')
plt.show()


if __name__ == "__main__":
    a()
    pass
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
from scipy.signal import find_peaks
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


def b():

    r = 2.0
    T_c = np.pi / (2.0*2.0)
    # epsi = 1e-3
    # T = T_c + epsi
    T = 1
    epsi = T - T_c  # Esto es un epsilon muy grande, preguntar

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


#
def calcAmplitud(n0=2, r=2.0):
    kk = solve(r=2.0, T=1.0, K=10.0,Nlim=100, N0=n0)
    # mean = kk[0].mean()
    rango = 1e-2
    maximo = kk[0].max()
    minimo = kk[0].min()

    peaks, _ = find_peaks(kk[0], height=maximo-rango)
    valleys, _ = find_peaks(-kk[0], height=-minimo-rango)
    # Tiro el primero porque suele tener un pelin menos de amplitud
    peaks = peaks[1:]
    valleys = valleys[1:]


    # plt.plot(kk[2], kk[0])
    # plt.plot(kk[2][peaks], kk[0][peaks], "rx")
    # plt.plot(kk[2][valleys], kk[0][valleys], "rx")
    # # plt.plot(np.zeros_like(kk[0]), "--", color="gray")
    # # plt.show()
    # # plt.pause(3)
    # plt.close()

    if len(peaks) < len(valleys):
        minlen = len(peaks)
    else:
        minlen = len(valleys)

    amplitud = []
    for i in range(minlen-1):

        amplitud.append(abs(kk[0][peaks[i]] - kk[0][valleys[i]]))
        amplitud.append(abs(kk[0][peaks[i]] - kk[0][valleys[i+1]]))
    
    # Ahora pasamos a calcular la frecuencia
    periodo1 = kk[2][peaks[1:]] - kk[2][peaks[:-1]]
    periodo2 = kk[2][valleys[1:]] - kk[2][valleys[:-1]]

    periodo = 0.5 * (periodo1.mean() + periodo2.mean())
    
    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    return np.array(amplitud).mean(), periodo


def barridoCI():
    valores = []
    n0 = np.linspace(1,20,20)
    n0 = n0[n0 != 10]
    for N0 in n0:

        if N0 == 10:
            continue

        # print(N0, end=' ')

        A, _ = calcAmplitud(N0)

        # print(A)

        valores.append(A)

    valores = np.array(valores)

    plt.plot(n0, valores, 'o')
    plt.show()

def barrido_r():

    rs = np.linspace(0.1,20,100)

    periodos = []

    for r in rs:

        _, periodo = calcAmplitud(n0=2, r=r)

        periodos.append(periodo)

    periodos = np.array(periodos)

    plt.plot(rs, periodos, 'o')
    plt.show()



if __name__ == "__main__":
    
    # a()

    # b()

    # barridoCI()

    barrido_r()

    pass
    
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
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

SAVE_PATH = "Figuras/Ej_01"

# CI = np.linspace(-3,2,16) # Con esto y K = 1 la cosa explota lindo
CI = np.linspace(-3, 2, 21)  # Las condiciones iniciales
K = 1  # El valor de K creo que no importa mucho
iteraciones = 15  # Cantidad de iteraciones
t = np.array(list(range(iteraciones + 1)))

# Funcion con la solucion analitica del modelo de Beverton-Holt
def BHEEcExacta(n0, t, r, K):
    num = K * n0
    den = n0 + (K - n0) * np.power(r, -t)
    return num / den


# Funcion para hacer una iteracion del modelo de Beverton-Holt
def BevertonHolt(n_t, r, K):
    return (r * n_t) / (1 + ((r - 1) / K) * n_t)


# Funcion para hacer un barrido en las condiciones iniciales, fijado r y K
def barridoEnCI(r, K):
    for n_0 in CI:
        n_t = np.array([n_0])

        for i in range(iteraciones):

            n_next = BevertonHolt(n_t[-1], 0.5, K)
            n_t = np.append(n_t, n_next)

        plt.plot(t, n_t, label=n_0)
    plt.tight_layout()
    plt.show()


def barridoEnCiExacto(r, K):
    for n_0 in CI:

        curva = BHEEcExacta(n_0, t, r, K)

        plt.plot(t, curva, label="n_0")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

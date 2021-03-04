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

# Constantes
CI = np.linspace(0, 2, 11)  # Las condiciones iniciales, solo con sentido si >=0
K = 1.0  # El valor de K creo que no importa mucho
iteraciones = 15  # Cantidad de iteraciones
t = np.array(list(range(iteraciones + 1)))

# Funcion con la solucion analitica del modelo de Beverton-Holt
def BHEEcExacta(n0, t, r, K):
    num = K * n0
    den = n0 + (K - n0) * np.power(r, -t)
    return num / den


# Funcion para hacer una iteracion del modelo de Beverton-Holt
def BevertonHolt(n_t, r, K):
    return (r * n_t) / (1 + ((r - 1.0) / K) * n_t)


# Funcion para hacer un barrido en las condiciones iniciales, fijado r y K
def barridoEnCI(r, K):

    for n_0 in CI:
        n_t = np.array([n_0])

        for i in range(iteraciones):

            n_next = BevertonHolt(n_t[-1], r, K)
            n_t = np.append(n_t, n_next)

        plt.plot(t, n_t, "--.", label=n_0)
    file_path = os.path.join(SAVE_PATH, "Mapeo_r={}".format(r))
    plt.xlabel("t")
    plt.ylabel(r"$n_{t}$")
    plt.tight_layout()
    plt.savefig(file_path, format="pdf")
    plt.close()

    # Ahora lo hago con la ec exacta, pero creo que es al pedo
    for n_0 in CI:
        curva = BHEEcExacta(n_0, t, r, K)
        plt.plot(t, curva, "--.", label="n_0")

    file_path = os.path.join(SAVE_PATH, "Exacta_r={}".format(r))
    plt.xlabel("t")
    plt.ylabel(r"$n(t)$")
    plt.tight_layout()
    plt.savefig(file_path, format="pdf")
    plt.close()


def coweb(r, K):
    n = np.linspace(0, 2, 1000)
    f_n = BevertonHolt(n, r, K)

    plt.plot(n, f_n, "-", label=r"$f(x)$")
    plt.plot([0, 2], [0, 2], label="Identidad")

    file_path = os.path.join(SAVE_PATH, "Coweb_r={}".format(r))
    plt.xlabel("x")
    plt.ylabel(r"$f(x)$")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(file_path, format="pdf")
    plt.close()


"""
Segun https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3759145/
r > 1 y K > 0
"""

if __name__ == "__main__":

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    rs = [1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 10.0]

    for r in rs:
        barridoEnCI(r, K)
        coweb(r, K)

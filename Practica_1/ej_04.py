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

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

SAVE_PATH = os.path.join("Figuras", "ej_04")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def ej4(Nt=20):

    N = np.array([1.0])

    for _ in range(1, Nt):
        r = np.random.poisson(1.7)
        n = r * N[-1]  # Calculamos la siguiente iteracion
        N = np.append(N, n)

    return N


plt.plot(ej4(), "bo")
plt.xlabel(r"$t$")
plt.ylabel(r"$N_{t}$")
plt.tight_layout()
file_name = os.path.join(SAVE_PATH, "simulacion.pdf")
plt.savefig(file_name, format="pdf")
plt.show()

count = 0
estadistica = 100000000
for i in range(estadistica):
    poisson = np.random.poisson(1.7, 20)
    if i % 1000000 == 0:
        print(i / 1000000)
    if 0 in poisson:
        count += 1
print("Porcentaje de casos en los que hay un cero en 20 iteraciones: {}".format(count/estadistica))
print("Porcentaje de casos en donde sobrevive la especie: {}".format(1-count/estadistica))
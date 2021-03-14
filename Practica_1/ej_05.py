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
import seaborn as sns
sns.set()

SAVE_PATH = os.path.join("Figuras", "ej_05")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Ecuacion del modelo logistico
def logisticModel(N,t,r,K):
    dNdt = r * N * (1 - N/K)
    return dNdt

# Funcion para obtener el tiempo de un desastre
def getDisasterTime(q,l):
    return -np.log(1-q) / l

# Definimos algunas constantes
K = 10      # El valor de K no importa mucho, es el valor al cual va a converger
N0 = K/2    # Condicion inicial

t_final = 250   # Cuanto tiempo vamos a iterar

# La relacion que tiene que cumplir r, p, y l para que la especie sobreviva es
# p * np.exp(r / l) > 1
r = 1            # Mas grande -> + rapido evolucina -> + chance de sobrevivir 
p = np.e**-1     # Fraccion que sobrevive
# Con r = 1 y p = 1/e la condicion para lambda (l) se reduce a
# l > 1
l = 0.95

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
plt.xlabel(r"$t$")
plt.ylabel(r"$N(t)$")
plt.tight_layout()
file_name = os.path.join(SAVE_PATH, "l={}.pdf".format(l))
plt.savefig(file_name, format='pdf')
plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 09-03-21
File: ej_06.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""
# TODO revisar lo de weak allee effect

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns

sns.set()

SAVE_PATH = os.path.join("Figuras", "Ej_06")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def modelAlleeEffect(N, t, r, K, A):
    dNdt = r * N * (1.0 - N / K) * (N / A - 1.0)
    return dNdt


def a(r=0.5, K=10, A=6):
    N0 = np.linspace(0, 15, 16)
    # N0 = np.linspace(0,15,31)
    t = np.linspace(0, 15, 10000)

    for n0 in N0:
        N = odeint(modelAlleeEffect, N0, t, args=(r, K, A))

        plt.plot(t, N, "--")

    plt.xlabel("t")
    plt.ylabel("N(t)")
    plt.tight_layout()
    plt.show()


def b(r=0.5, K=10, A=6):
    N = np.linspace(0, K+1, 10000)
    dNdt = modelAlleeEffect(N, 0, r, K, A)

    plt.plot(N, dNdt)
    plt.scatter([0, K, A], [0, 0, 0], s=80, facecolors="none", edgecolors="r")
    plt.scatter([0, K], [0, 0], s=80, facecolors="r", edgecolors="r")
    # plt.arrow(0, 0, A / 2, 0, head_width=0.02, head_length=0.5, fc="k", ec="k")
    plt.xlabel(r"$N(t)$")
    plt.ylabel(r"$\dot{N}(t)$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    a()

    b()
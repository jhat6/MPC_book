# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:04:00 2020

Water Heater Control, Section 3.4 Case Study in Model Predictive Control by 
Camacho and Bordons

@author: jeffa
"""


import control
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Plant model
sys = control.tf([0.2713],[1, -0.8352, 0 , 0], 1)

# Generate step response data
tSteps = np.linspace(0, 50, 51)
u = np.ones(51)
T, stepData = control.step_response(sys, tSteps)
plt.plot(T, stepData)

# Set model, prediction, and control horizons
Nm = 30
Np = 10
Nc = 5

# G matrix
G = np.tril(sp.linalg.toeplitz(stepData[1:Np+1]))
G = G[:, 0:Nc]
GT = np.transpose(G)

# controller gain lambda parameter
lam = 1

# Calculate the controller gain matrix
K_mat, *_ = np.linalg.lstsq(GT.dot(G)+lam*np.eye(Nc), GT, rcond=None)
# Use first row of the gain matrix
K = K_mat[0,:].reshape(1, Np)

# Calculate matrix F for the free response f
F = sp.linalg.hankel(stepData[2:Np+2], stepData[Np:Nm]) - \
    np.tile(stepData[1:Nm-Np+1], (Np,1))

# Past calculated control moves
Upast = np.zeros(Nm-Np)

cycles = 120

# set the reference trajectory
ref = np.ones((cycles+Np,))
ref[30:60] = 0
ref[90:] = 0

u = 0   # initial control move
# Arrays to store calculated input and output
u_cont = np.zeros((cycles,))
ym = np.zeros((cycles+3,))
# Simulate the closed-loop system
for k in range(0, cycles):
    f = ym[k] + F.dot(Upast)
    w = ref
    deltaU = K.dot(w[k:k+Np]-f)
    deltaU = K.dot(w[k]*np.ones(Np,)-f)
    u += deltaU        
    Upast = np.concatenate((deltaU, Upast[:-1]))
    u_cont[k] = u
    ym[k+3] = 0.8352*ym[k+2] + 0.2713*u


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(ym)
ax1.plot(w)
ax2.plot(u_cont)

    



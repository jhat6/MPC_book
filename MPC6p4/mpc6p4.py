# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:53:01 2020

@author: jeffa

Based on Example Flight Control System Case Study, Section 6.4 of 
Model Predictive by Camacho and Bordons

"""

import numpy as np
import mpc_ss_mats 
import matplotlib.pyplot as plt
#import time 

# Discrete time LTI plant matrices (Ts = 0.1 seconds)
Am = np.matrix('0.996, 0.0383, 0.0131, -0.0322; \
               -0.0056, 0.9647, 0.7446, 0.0001; \
               0.002, -0.0097, 0.9543, 0;       \
               0.001, -0.0005, 0.0978, 1')

Bm = np.matrix('0.0001, 0.1002; \
               -0.0615, 0.0183; \
               -0.1133, 0.0586; \
               -0.0057, 0.0029')
               
Cm = np.matrix('1, 0, 0, 0; \
                0, -1, 0, 7.74')
                
# Prediction horizon                
Np = 30
# Control horizon
Nc = 10
# Penalty on the control action
R = np.diag((5,5))
# Penalty on setpoint deviation
Q = np.diag((1,1))

# number of system states
m = np.shape(Am)[0]
# number of outputs
q = np.shape(Cm)[0]
# Number of inputs
p = np.shape(Bm)[1]

# Calculate prediction matrics F, Phi, Cbar and velocity form 
# state space matrics A, B, C
F, Phi, Cbar, A, B, C = mpc_ss_mats.sparse_mats(Am, Bm, Cm, Np, Nc)

# Cast output and control penalties over prediction and control 
# horizons, resp.
Qbar = np.zeros((q*Np, q*Np))
Rbar = np.zeros((p*Nc, p*Nc))
for i in range(0, Nc):    
    Rbar[p*i:p*(i+1), p*i:p*(i+1)] = R

for i in range(0, Np):    
    Qbar[q*i:q*(i+1), q*i:q*(i+1)] = Q

# Compute the optimal state feedback control gains
Ks1 = Phi.T*Cbar.T*Qbar*Cbar*Phi+Rbar
Ks2 = Phi.T*Cbar.T*Qbar*Cbar*F
Rs1 = Phi.T*Cbar.T*Qbar
Ks = np.linalg.inv(Ks1)*Ks2     # Optimal FB gain
Kx = Ks[0:p, 0:m]               # Optimal FB gain for the state variables
Ky = Ks[0:p, m:]                # Optimal FB gain for the outputs
Rs = np.linalg.inv(Ks1)*Rs1     # Optimal setpoint FF gain

# Examine eigenvalues of the closed-loop system matrix
Ac = A-B*Ks[0:p, :]
lm, _ = np.linalg.eig(Ac)
print("Closed-loop system eigenvalues")
print(lm)

# number of samples to simulate
cycles = 200    

# Create setpoints for the outputs: [10; 0] for the first 100 samples and 
# [10; 5] for the next 100 samples
temp = np.ones((Np, int(cycles/2)))
r1 = np.matrix('10; 0')
r1 = np.kron(temp, r1)
r2 = np.matrix('10; 5')
r2 = np.kron(temp, r2)
r = np.hstack((r1, r2))

# Setup simulation variables
xm = np.matrix(np.zeros((m, cycles)))
y = np.matrix(np.zeros((q, cycles)))
u = np.matrix(np.zeros((p, cycles)))

    
#%% Simulation for the plant in normal form (delta state and control)
x = np.matrix(np.zeros((m+q, cycles)))    
Delu = np.matrix(np.zeros((p, cycles)))    
for k in range(0, cycles-1):        
    Delu[:, k] = -Kx*x[0:m, k] -Ky*y[:, k] + Rs[0:p, :]*r[:, k]
    x[0:m, k+1] = Am*x[0:m, k] + Bm*Delu[:, k]
    y[:, k+1] = Cm.dot(Am)*x[0:m, k] + y[:, k] + Cm.dot(Bm)*Delu[:, k]
    
# Output for plotting
u_cont = Delu
    
#%% Simulation for the plant in normal form (normal state and control)
x = np.matrix(np.zeros((m+q, cycles)))    
Delu = np.matrix(np.zeros((p, cycles)))    
for k in range(0, cycles-1):        
    if k <1:
        u[:, k] = -Kx*x[0:m, k] -Ky*y[:, k] + Rs[0:p, :]*r[:, k]
    else:
        u[:, k] = u[:, k-1] -Kx*(x[0:m, k]-x[0:m, k-1]) -Ky*y[:, k] + Rs[0:p, :]*r[:, k]
    x[0:m, k+1] = Am*x[0:m, k] + Bm*u[:, k]
    y[:, k+1] = Cm*x[0:m, k+1] 
    
# # Output for plotting
u_cont = u

    
#%% Simulation for the plant in velocity form

# x = np.matrix(np.zeros((m+q, cycles)))    
# Delu = np.matrix(np.zeros((p, cycles)))    
# for k in range(0, cycles-1):    
#     Delu[:, k] = -Ks[0:p,:]*x[:, k] + Rs[0:p, :]*r[:, k]
#     x[:, k+1] = A*x[:, k] + B*Delu[:, k]
#     y[:, k] = C*x[:, k]    

# # Output for plotting
# u_cont = Delu

#%% Plot results
fig, ax = plt.subplots(3,1)
ax[0].plot(r[0, 0:-1].T, '--')
ax[0].plot(y[0, 0:-1].T)
ax[0].set_ylabel('Output 1')

ax[1].plot(r[1, 0:-1].T, '--')
ax[1].plot(y[1, 0:-1].T)
ax[1].set_ylabel('Output 2')

ax[2].plot(u_cont[0,0:-1].T)
ax[2].plot(u_cont[1,0:-1].T,'r')
ax[2].set_ylabel('Control Action')
ax[2].set_xlabel('Samples')

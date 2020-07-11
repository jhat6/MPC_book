# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:59:17 2020


Based on Example Flight Control System Case Study, Section 6.4 of 
Model Predictive by Camacho and Bordons

Constraints added

"""

import numpy as np
import mpc_ss_mats 
import matplotlib.pyplot as plt
import time 
import osqp
import scipy as sp

start = time.time()
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

# number of samples to simulate
cycles = 200    

# Create setpoints for the outputs: [10; 0] for the first 100 samples and 
# [10; 5] for the next 100 samples
temp = np.ones((Np, int(cycles/2)))
r1 = np.matrix('5; 0')
r1 = np.kron(temp, r1)
r2 = np.matrix('5; 2')
r2 = np.kron(temp, r2)
r = np.hstack((r1, r2))

# Setup simulation variables
y = np.matrix(np.zeros((q, cycles)))
u = np.matrix(np.zeros((p, cycles)))
    
#%% Simulation for the plant in velocity form

x = np.matrix(np.zeros((m+q, cycles)))    
Delu = np.matrix(np.zeros((p, cycles)))    

# Setup QP optimization: min (x*Px + x*q)
P_opt = sp.sparse.csc_matrix(Phi.T*Cbar.T*Qbar*Cbar*Phi + Rbar)
q_opt = Phi.T*Cbar.T*Qbar*(Cbar*F*x[:, 0] - r[:, 0])
# set constraints
# control delta constraints
DelUhi = np.matrix(2*np.ones((p*Nc,1)))
DelUlo = np.matrix(-2*np.ones((p*Nc,1)))
A_cons = np.eye(p*Nc)
# output constraints
yhi = 5
ylo = -5
Yhi = np.matrix(yhi*np.ones((q*Np,1))) - Cbar*F*x[:, 0]
Ylo = np.matrix(ylo*np.ones((q*Np,1))) - Cbar*F*x[:, 0]

uineq = np.vstack((DelUhi, Yhi))
lineq = np.vstack((DelUlo, Ylo))
A_cons = sp.sparse.vstack((A_cons, Cbar*Phi), format='csc')

# Initialize the QP problem
prob = osqp.OSQP()
prob.setup(P_opt, q_opt, A=A_cons, l=lineq, u=uineq, warm_start=True, verbose=False)

for k in range(0, cycles-1):                
    # solve optimization to compute the control delta
    res = prob.solve()    
    Delu[:,k]=np.matrix(res.x[0:2]).T
    # Simulate the plant
    x[:, k+1] = A*x[:, k] + B*Delu[:, k]
    y[:, k] = C*x[:, k]    
    # update optimization with the current state and reference
    q_opt = Phi.T*Cbar.T*Qbar*(Cbar*F*x[:, k] - r[:,k])
    Yhi = np.matrix(yhi*np.ones((q*Np,1))) - Cbar*F*x[:,k]
    Ylo = np.matrix(ylo*np.ones((q*Np,1)))- Cbar*F*x[:,k]
    uineq = np.vstack((DelUhi, Yhi))
    lineq = np.vstack((DelUlo, Ylo))
    prob.update(q=q_opt, l=lineq, u=uineq)
    
    
# Output for plotting
u_cont = Delu
print(time.time()-start)
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

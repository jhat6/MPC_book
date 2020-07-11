'''
Taken from:    
    Tutorial Workshop on Model Predictive Control
    ACC 2007, New York
    Liuping Wang
'''

import numpy as np
import mpc_ss_mats  
                
Am = np.matrix('1, 1; 0, 1')
Bm = np.matrix('0.5; 1')
Cm = np.matrix('1, 0')
                
Np = 30
Nc = 3
R = 1
Q = 1
q = 1
p = 1
Qbar = np.zeros((q*Np, q*Np))
Rbar = np.zeros((p*Nc, p*Nc))
for i in range(0, Nc):    
    Rbar[p*i:p*(i+1), p*i:p*(i+1)] = R

for i in range(0, Np):    
    Qbar[q*i:q*(i+1), q*i:q*(i+1)] = Q
    
F, Phi, Cbar, *_ = mpc_ss_mats.mats(Am, Bm, Cm, Np, Nc)

CbarPhi = Cbar.dot(Phi)
Ks1 = CbarPhi.T.dot(Qbar).dot(CbarPhi)+Rbar
Ks2 = CbarPhi.T.dot(Qbar).dot(Cbar).dot(F)
Rs1 = CbarPhi.T.dot(Qbar)
Ks = np.linalg.inv(Ks1).dot(Ks2)
Rs = np.linalg.inv(Ks1).dot(Rs1)
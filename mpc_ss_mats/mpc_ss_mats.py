# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:36:12 2020

@author: jeffa

Creates matrices for state space model predictive control.  Two available 
functions for normal matrices and sparse matrices

Inputs:
        State space matrices Am, Bm, Cm:
            
            xm(k+1) = Am*xm(k) + Bm*u(k)
            y(k) = Cm*xm(k)
            
        Np: Prediction horizon
        Nc: Control horizion
        
Outputs:
        State space matrices Cbar, F, Phi cast over the prediction 
        and control horizons:
            
        Y = Cbar*F*x(k) + Cbar*Phi*DelU
        Plant matrices in velocity form:
        x(k+1) = A*x(k) + B*Delu(k)
        y(k) = C*x(k)
        x(k) = [Delxm(k); y(k)]
        Delxm(k) = xm(k) - xm(k-1)
        Delu(k) = u(k) - u(k-1)
        
"""

#%%
def mats(Am, Bm, Cm, Np, Nc):

    import numpy as np    
    
    # Create Velocity form matrices
    
    m = np.shape(Am)[0] # number of states
    q = np.shape(Cm)[0] # number of outputs
    p = np.shape(Bm)[1] # number of inputs
    A = np.block([[Am, np.zeros((m, q))], 
                  [Cm.dot(Am), np.eye(q)]])
    B = np.block([[Bm], [Cm.dot(Bm)]])
    C = np.block([np.zeros((q, m)), np.eye(q)])
    
    # Predict Np steps into the future 

    m1 = m+q
    F = np.matrix(np.empty((Np*(m1), m1)))    
    Phi = np.matrix(np.zeros((Np*(m1), Nc*p)))                
    Cbar = np.kron(np.eye(Np), C)
    
    for i in range(0, Np):
        F[m1*i:m1*(i+1), :] = A**(i+1)
        for j in range(0, min(i+1, Nc)):            
            Phi[m1*i:m1*(i+1), p*j:p*(j+1)] = A**(i-j)*B
    
    return F, Phi, Cbar, A, B, C

#%%
def sparse_mats(Am, Bm, Cm, Np, Nc):
    import numpy as np
    from scipy import sparse
    
    # Create Velocity form matrices
    
    m = np.shape(Am)[0] # number of states
    q = np.shape(Cm)[0] # number of outputs
    p = np.shape(Bm)[1] # number of inputs
    A = np.block([[Am, np.zeros((m, q))], 
                  [Cm.dot(Am), np.eye(q)]])
    B = np.block([[Bm], [Cm.dot(Bm)]])
    C = np.block([np.zeros((q, m)), np.eye(q)])
    
    # Predict Np steps into the future 

    m1 = m+q
    F = sparse.lil_matrix(np.empty((Np*(m1), m1)))    
    Phi = np.matrix(np.zeros((Np*(m1), Nc*p)))     
    Cbar = sparse.kron(sparse.eye(Np), C)
    
    for i in range(0, Np):
        F[m1*i:m1*(i+1), :] = A**(i+1)    
        for j in range(0, min(i+1, Nc)):            
            Phi[m1*i:m1*(i+1), p*j:p*(j+1)] = A**(i-j)*B
    
    return F, Phi, Cbar, A, B, C

#%%
def ctrl_mats(Am, Bm, Cm, Np, Nc, MV, PV):
    '''
        Uses non-sparse matrices for prediction model
    '''
    import numpy as np    
        
    # Create Velocity form matrices - state space plant for control calc
    Ac = Am
    Bc = Bm[:, MV]
    Cc = Cm[PV, :]
    
    m = np.shape(Ac)[0] # number of states
    q = np.shape(Cc)[0] # number of PVs for feedback
    p = np.shape(Bc)[1] # number of MVs for control calculation
        
    A = np.block([[Ac, np.zeros((m, q))], 
                  [Cc.dot(Ac), np.eye(q)]])
    B = np.block([[Bc], [Cc.dot(Bc)]])
    C = np.block([np.zeros((q, m)), np.eye(q)])        
    
    # Predict Np steps into the future 
    m1 = m+q   
    F = np.matrix(np.empty((Np*(m1), m1)))                 
    Phi = np.matrix(np.zeros((Np*(m1), Nc*p)))
    Cbar = np.kron(np.eye(Np), C)
    
    for i in range(0, Np):
        F[m1*i:m1*(i+1), :] = A**(i+1)    
        for j in range(0, min(i+1, Nc)):
            Phi[m1*i:m1*(i+1), p*j:p*(j+1)] = A**(i-j)*B
    
    return F, Phi, Cbar, A, B, C

#%%     
def sparse_ctrl_mats(Am, Bm, Cm, Np, Nc, MV, PV):
    '''
        Uses sparse matrices for prediction model
    '''
    import numpy as np
    from scipy import bmat
    from scipy import sparse
        
    # Create Velocity form matrices - state space plant for control calc
    Ac = Am
    Bc = Bm[:, MV]
    Cc = Cm[PV, :]
    
    m = np.shape(Ac)[0] # number of states
    q = np.shape(Cc)[0] # number of PVs for feedback
    p = np.shape(Bc)[1] # number of MVs for control calculation
        
    A = bmat([[Ac, np.zeros((m, q))], 
                  [Cc.dot(Ac), np.eye(q)]])
    B = bmat([[Bc], [Cc.dot(Bc)]])
    C = bmat([np.zeros((q, m)), np.eye(q)])        
    
    # Predict Np steps into the future 
    m1 = m+q
    F = sparse.csr_matrix(np.empty((Np*(m1), m1)))             
    Phi = sparse.csr_matrix(np.zeros((Np*(m1), Nc*p)))     
    Cbar = sparse.kron(sparse.eye(Np), C)    
    
    for i in range(0, Np):
        F[m1*i:m1*(i+1), :] = A**(i+1)    
        for j in range(0, min(i+1, Nc)):            
            Phi[m1*i:m1*(i+1), p*j:p*(j+1)] = A**(i-j)*B
    
    return F, Phi, Cbar, A, B, C
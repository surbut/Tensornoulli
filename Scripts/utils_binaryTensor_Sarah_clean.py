# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:41:51 2024

@author: chyga
"""

import numpy as np
from scipy.special import expit, logit

class TensorModelWithGenetics:
    def __init__(self, N, P, D, K, R1, R2, T):
        self.N = N  # Number of individuals
        self.P = P  # Number of genetic variants
        self.D = D  # Number of diseases
        self.K = K  # Number of signatures
        self.R1 = R1  # Rank for individual patterns
        self.R2 = R2  # Rank for disease patterns
        self.T = T  # Number of time points
        self.initialized = False
        self.initialize_parameters()

        
    def initialize_parameters(self):
        self.U1 = np.random.randn(self.N, self.K, self.R1) * 0.5  # Increased from 0.01
        self.U2 = create_smooth_basis(self.T, self.R1)
        self.W = np.random.randn(self.D, self.K, self.R2) * 0.5  # Increased from 0.01
        self.U3 = create_smooth_basis(self.T, self.R2)
        self.G = np.random.binomial(2, 0.3, size=(self.N, self.P))
        self.B = np.random.randn(self.P, self.K, self.R1) * 0.05  # Increased from 0.001
        self.C = np.random.randn(self.N, self.K, self.R1) * 0.05  # Increased from 0.001
        self.initialized = True
 
    def get_dimensions(self):
        return {
            'N': self.N,
            'P': self.P,
            'D': self.D,
            'K': self.K,
            'R1': self.R1,
            'R2': self.R2,
            'T': self.T
        }
    def get_parameters(self):
        if not self.initialized:
            raise ValueError("Parameters have not been initialized yet.")
        return {
        'U1': self.U1,
        'U2': self.U2,
        'W': self.W,
        'U3': self.U3,
        'G': self.G,
        'B': self.B,
        'C': self.C
    }
    

    def compute_U1G(self):
        U1G = self.U1.copy()
        genetic_effect = np.einsum('np,pkr->nkr', self.G, self.B)
        U1G = genetic_effect #+ self.C
        return U1G

    def compute_theta(self):
        U1G = self.compute_U1G()
        lambda_k = np.einsum('nkr,tr->nkt', U1G, self.U2)
        phi = np.einsum('dkr,tr->dkt', self.W, self.U3)
        theta = np.einsum('nkt,dkt->ndt', lambda_k, phi)
        return theta

    def survival_likelihood(self, Y, S):
        theta = self.compute_theta()
        pi = expit(theta)
        log_likelihood = 0
        for n in range(self.N):
            for d in range(self.D):
                t = S[n, d]
                log_likelihood += np.sum(np.log(1 - pi[n, d, :t] + 1e-10))
                if Y[n, d, t] == 1:
                    log_likelihood += np.log(pi[n, d, t] + 1e-10)
                    pi[n, d, t+1:] = 0
        return log_likelihood

    def compute_gradients(self, Y, S):
        theta = self.compute_theta()
        pi = expit(theta)
        U1G = self.compute_U1G()
        lambda_k = np.einsum('nkr,tr->nkt', U1G, self.U2)
        phi = np.einsum('dkr,tr->dkt', self.W, self.U3)
        
        d_theta = np.zeros_like(theta)
        for n in range(self.N):
            for d in range(self.D):
                t = S[n, d]
                d_theta[n, d, :t] = -pi[n, d, :t]
                if Y[n, d, t] == 1:
                    d_theta[n, d, t] += 1
        
        d_lambda = np.einsum('ndt,dkt->nkt', d_theta, phi)
        d_phi = np.einsum('ndt,nkt->dkt', d_theta, lambda_k)
        
        d_U1G = np.einsum('nkt,tr->nkr', d_lambda, self.U2)
        d_W = np.einsum('dkt,tr->dkr', d_phi, self.U3)
        
        d_U1 = d_U1G
        d_B = np.einsum('nkr,np->pkr', d_U1G, self.G)
        d_C = d_U1G
        
        return d_U1, d_W, d_B, d_C

    def fit(self, Y, S, num_iterations=1000, learning_rate=1e-4, l2_reg=1e-5, clip_value=1.0, patience=50):
        losses = []
        best_loss = -np.inf
        patience_counter = 0
        
        for iteration in range(num_iterations):
            log_likelihood = self.survival_likelihood(Y, S)
            loss = log_likelihood - l2_reg * (np.sum(self.U1**2) + np.sum(self.W**2) + np.sum(self.B**2) + np.sum(self.C**2))
            losses.append(loss)
            
            if loss > best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iteration}")
                    break
            
            d_U1, d_W, d_B, d_C = self.compute_gradients(Y, S)
            
            d_U1 = np.clip(d_U1, -clip_value, clip_value)
            d_W = np.clip(d_W, -clip_value, clip_value)
            d_B = np.clip(d_B, -clip_value, clip_value)
            d_C = np.clip(d_C, -clip_value, clip_value)
            
            self.U1 += learning_rate * (d_U1 - 2 * l2_reg * self.U1)
            self.W += learning_rate * (d_W - 2 * l2_reg * self.W)
            self.B += learning_rate * (d_B - 2 * l2_reg * self.B)
            self.C += learning_rate * (d_C - 2 * l2_reg * self.C)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Log-likelihood: {log_likelihood}")
        
        return losses
    

# Set random seed for reproducibility
np.random.seed(42)

def create_smooth_basis(T, R):
    t = np.linspace(0, 1, T)
    basis = np.zeros((T, R))
    basis[:, 0] = 4 * (1 - t)**3  # early peaking
    basis[:, 1] = 27 * t * (1 - t)**2  # middle peaking
    basis[:, 2] = 4 * t**3  # late peaking
    return basis
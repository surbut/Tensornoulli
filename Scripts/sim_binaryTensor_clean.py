# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:48:00 2024

@author: chyga
"""
from scipy.special import expit
import numpy as np
import tensorly as tl
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from utils_binaryTensor_Sarah import TensorModelWithGenetics



# set up the parameter 
# generate the outcomes with Sarah's model
## N is the dimension of subjects
## P is the dimension of X
## D is the dimension of disease
## K is the dimension of topics (or factors)
## R1 and R2 are the dimensions of basis functions
N, P, D, K, T = 1000, 20, 10, 5, 10
R1 = R2 = 3
np.random.seed(2)
# generate time points 
T_vec = np.linspace(0, 1, T)
# generate trend function
# h_1 = np.sin(0.2 * np.pi * T_vec)
# h_2 = T_vec * (10 - T_vec)/10
# h_3 = np.cos(0.2 * np.pi * T_vec) + 1
h_1 = 4 * (1 - T_vec)**3  # early peaking
h_2 = 27 * T_vec * (1 - T_vec)**2  # middle peaking
h_3 = 4 * T_vec**3  # late peaking
Phi_T = np.vstack((h_1, h_2, h_3)).T
Phi_T, _ = np.linalg.qr(Phi_T)
# # generate B1 and B2 
# B1 = np.random.normal(size = (N, R1, 3))
# B2 = np.random.normal(size = (K, R2, 3))

# A1 = tl.tenalg.mode_dot(B1, Phi_T, 2)
# A2 = tl.tenalg.mode_dot(B2, Phi_T, 2)
true_model = TensorModelWithGenetics(N, P, D, K, R1, R2, T)
true_model.initialize_parameters()
theta_true = true_model.compute_theta()
X = true_model.G # the genetic covariates for each individual
# theta_true = np.einsum('irt, jrt -> ijt', 
#           A1, A2)

type_outcome = 'Binary'

if type_outcome == 'Binary':
    # generate outcomes for Gaussian
    error = np.random.normal(scale = 1, loc = 0, size = (N, D, T))
    Y = theta_true + error
if type_outcome == 'Binary':
    # generate binary outcomes
    error = np.random.logistic(scale = 1, loc = 0, size = (N, D, T))
    Y = ((theta_true + error)>0).astype('int')


# begin our estimation
A1, w, A2 = svds(Y.mean(axis = 2), k = K) # average over time
# A, D, G = svds(theta_pred_glm.mean(axis = 2), k = R) # average over time
A1 = A1 @ np.diag(np.sqrt(w))
A2 = np.diag(np.sqrt(w)) @ A2
A2 = A2.T
# # repeat for P-degree of polynomials of time
# P = 10
# # construct 10th order Legendre polynomial  basis function for the time mode
# def basis(n, x): 
#     if(n == 0):
#         return 1 # P0 = 1
#     elif(n == 1):
#         return x # P1 = x
#     else:
#         return (((2 * n)-1)*x * basis(n-1, x)-(n-1)* basis(n-2, x))/float(n)

# Phi_G = []
# # Phi_G.append(
# #    np.ones(shape=(T_train))/T)
# for deg in range(1, (P+1)):
#     Phi_G.append( basis(deg, T_vec))

# Phi_G = np.column_stack(Phi_G)
# Phi_G, _ = np.linalg.qr(Phi_G)

# # Phi_A equals to Phi_G
# Phi_A = Phi_G
# the basis functions are the truth
Phi_A1 = Phi_A2 = Phi_T
## polynomial basis
## the varying of loading should be larger
# Phi = np.column_stack((np.sin(np.array(range(T_train)) * np.pi/T),
#                        np.array(range(T_train))/T,
#                        np.cos(np.array(range(T_train)) * np.pi/T) + 1
#                        ))
# project A1 onto the space spanned by the time bases
B1 = tl.tenalg.mode_dot(np.repeat(A1[:, :, np.newaxis], T, axis = 2),
                       np.linalg.pinv(Phi_A1.T @ Phi_A1) @ Phi_A1.T, mode = 2)
# project B1 onto the space spanned by the covariates
B1 = tl.tenalg.mode_dot(B1, X @ np.linalg.pinv(X.T @ X) @ X.T, mode = 0)

# project A2 onto the space spanned by the time bases
B2 = tl.tenalg.mode_dot(np.repeat(A2[:, :, np.newaxis], T, axis = 2),
                       np.linalg.pinv(Phi_A2.T @ Phi_A2) @ Phi_A2.T, mode = 2)
# A = np.repeat(A[:, :, np.newaxis], Phi_A.shape[1], axis = 2)
# G = np.repeat(G[:, :, np.newaxis], Phi_G.shape[1], axis = 2)
# D is the diagnoal term
# D_T = np.repeat(D[:, np.newaxis], T_train, axis = 1)
# begin gradient descent for optimization
stepsize = .1
alpha = 0.5; beta = 0.8 # backtracking line search
rho = 1
stepsize_B1 = stepsize_B2 = stepsize
# the time is the last mode (i.e, t_idx = 1)
t_idx = len(Y.shape) - 1
niters = 1000
for it in range(niters):
    # A1 is time-varying loaded with genetic covariates
    A1_T = tl.tenalg.mode_dot(B1, Phi_A1, 2)
    # A2 is time-varying loaded
    A2_T = tl.tenalg.mode_dot(B2, Phi_A2, 2)
    
    theta_pre = np.einsum("irt, jrt -> ijt", 
                          A1_T, A2_T)
    
    if type_outcome == 'Gaussian':
        L_nabula = -Y + theta_pre
    
    if type_outcome == 'Binary':
        L_nabula = -Y + expit(theta_pre)
    
    # unfold along the first mode at each time
    L_nabula_mode0 = np.stack([tl.unfold(L_nabula[:, :, idx], mode = 0) for 
                                     idx in range(L_nabula.shape[t_idx])],
                                    axis = t_idx)
    L_nabula_mode1 = np.stack([tl.unfold(L_nabula[:, :, idx], mode = 1) for 
                                     idx in range(L_nabula.shape[t_idx])],
                                    axis = t_idx)
    
    # record the loss function pre-GD
    if type_outcome == 'Gaussian':
        loss_pre = np.sum((Y - theta_pre)**2)/N
    
    if type_outcome == 'Binary':
        loss_pre = (- Y * theta_pre + np.log(1 + np.exp(theta_pre))).sum()/N
    
    
    # grad_f_X_unflod_mode3 = tl.unfold(L_nabula, mode = 2)
    grad_B2 = tl.tenalg.mode_dot(np.einsum('ijt, jrt -> irt', 
                                          L_nabula_mode1, A1_T),
                                Phi_A2.T, 2) #+ rho * B2
    
    B2_tilde = B2 - stepsize_B2 * grad_B2
    
    # record the loss function post-GD
    if type_outcome == 'Gaussian':
        loss_after = np.sum((Y - np.einsum("irt, jrt -> ijt", A1_T, 
                              tl.tenalg.mode_dot(B2_tilde, Phi_A2, 2)))**2)/N
    
    if type_outcome == 'Binary':
        loss_after = (- Y * np.einsum("irt, jrt -> ijt", A1_T, 
                              tl.tenalg.mode_dot(B2_tilde, Phi_A2, 2)) +\
                      np.log(1 + np.exp(np.einsum("irt, jrt -> ijt", A1_T, 
                                            tl.tenalg.mode_dot(B2_tilde, Phi_A2, 2))))).sum()/N
    
    
    
    if(loss_after > loss_pre - alpha * \
        stepsize_B2 * 1/np.linalg.norm(grad_B2)):
        # return to previous stage
        stepsize_B2 *= beta
    else:
        B2 = B2_tilde
        A2_T = tl.tenalg.mode_dot(B2, Phi_A2, 2)
        # scale the G_T
        
    
    # grad for U1
    # X.T @ 
    grad_B1 = tl.tenalg.mode_dot(np.einsum('ijt, jrt -> irt', 
                                          L_nabula_mode0, A2_T),
                                Phi_A1.T, 2) #+ rho * B1
    
    B1_tilde = B1 - stepsize_B1 * grad_B1
    B1_tilde = tl.tenalg.mode_dot(B1_tilde, X @ np.linalg.pinv(X.T @ X) @ X.T, mode = 0)

    # record the loss function post-GD
    ## with back-track line search
    if type_outcome == 'Gaussian':
        loss_after = np.sum((Y - np.einsum("irt, jrt -> ijt", 
                                            tl.tenalg.mode_dot(B1_tilde, Phi_A1, 2), 
                                            A2_T))**2)/N
    
    if type_outcome == 'Binary':
        loss_after = (- Y * np.einsum("irt, jrt -> ijt", 
                                            tl.tenalg.mode_dot(B1_tilde, Phi_A1, 2), 
                                            A2_T) + \
                      np.log(1 + np.exp(np.einsum("irt, jrt -> ijt", 
                                                          tl.tenalg.mode_dot(B1_tilde, Phi_A1, 2), 
                                                          A2_T)))).sum()/N
    
    
    
    if(loss_after > loss_pre - alpha * \
        stepsize_B1 * 1/np.linalg.norm(grad_B1)):
        # return to previous stage
        stepsize_B1 *= beta
    else:
        B1 = B1_tilde
        A1_T = tl.tenalg.mode_dot(B1, Phi_A1, 2)
    
    theta_after = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
    
    if type_outcome == 'Gaussian':
        loss_after = np.sum((Y - theta_after)**2)/N
    
    if type_outcome == 'Binary':
        loss_after = (- Y * theta_after + np.log(1 + np.exp(theta_after))).sum()/N
                                                                                  
    # record the relative loss change
    tol_temp = np.abs(np.linalg.norm(loss_pre) - np.linalg.norm(loss_after))/\
        np.linalg.norm(loss_pre)
    
    if not (it % 100):
        print(f'{it}th iteration with loss: {np.round(loss_after,3)}')
        # print()
    if tol_temp < 1e-5 and (stepsize_B1 < 1e-6 or stepsize_B2 < 1e-6):
        print('Stop: not enough improvement')
        break

# theta_pred_glm = get_theta_binary_stratified(Y = Y,
#                                   X = np.ones((N, 1)))
# loss_glm = (- Y * theta_pred_glm + np.log(1 + np.exp(theta_pred_glm))).sum()/N
# loss_glm
loss_after
# time-varying loadings
A2_T = tl.tenalg.mode_dot(B2, Phi_A2, 2)  
A1_T = tl.tenalg.mode_dot(B1, Phi_A1, 2)  
theta_fit = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
# scale the time-varying loadings at each time
# for t in range(T_train):
#     G_T[:, :, t] = abs(G_T[:, :, t])/ \
#         np.sqrt((G_T[:, :, t]**2).mean(axis = 0))[np.newaxis, :]
# time-varying loadings
# plt.plot((G_T[:,0,:]).T/(G_T[:,0,:]).mean(),
#          color = '#66C2A5') 
# plt.plot((G_T[:,1,:]).T/(G_T[:,1,:]).mean(),
#          color = '#FC8D62') 
# plt.plot((G_T[:,2,:]).T/(G_T[:,2,:]).mean(),
#          color = '#8DA0CB') 
pi_true = expit(theta_true)
pi_fit = expit(theta_fit)
# Plot true vs estimated pi for a few randomly selected individuals and diseases
num_samples = 5
sample_individuals = np.random.choice(N, num_samples, replace=False)
sample_diseases = np.random.choice(D, num_samples, replace=False)

plt.figure(figsize=(15, 15))
for i, (n, d) in enumerate(zip(sample_individuals, sample_diseases)):
    plt.subplot(num_samples, 2, 2*i + 1)
    plt.plot(pi_true[n, d, :], label='True')
    plt.plot(pi_fit[n, d, :], label='Estimated')
    plt.title(f'Individual {n}, Disease {d}')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    
    plt.subplot(num_samples, 2, 2*i + 2)
    plt.scatter(pi_true[n, d, :], pi_fit[n, d, :])
    plt.plot([0, 1], [0, 1], 'r--')  # diagonal line
    plt.xlabel('True Probability')
    plt.ylabel('Estimated Probability')
    plt.title(f'True vs Estimated (Ind {n}, Disease {d})')

plt.tight_layout()
plt.show()
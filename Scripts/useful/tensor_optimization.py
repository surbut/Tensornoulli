import numpy as np
import tensorly as tl
from scipy.special import expit
from utils_binaryTensor_Sarah import survival_likelihood
from scipy.special import legendre


def generate_polynomial_bases(T, K):
    T_vec = np.linspace(0, 1, T)
    Phi = np.zeros((T, K))
    for k in range(K):
        Phi[:, k] = legendre(k)(2 * T_vec - 1)
    Phi, _ = np.linalg.qr(Phi)
    return Phi

def create_theta(Phi_A1,Phi_A2,B1_optimized,B2_optimized):
    A2_T = tl.tenalg.mode_dot(B2_optimized, Phi_A2, 2)  
    A1_T = tl.tenalg.mode_dot(B1_optimized, Phi_A1, 2)  
    theta_fit = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
    return theta_fit,A1_T,A2_T

def optimize_tensor_model(Y, event_times, X, B1, B2, Phi_A1, Phi_A2, type_outcome, max_iter=500, tol=1e-5):
    N, D, T = Y.shape
    stepsize = 0.1
    alpha = 0.5
    beta = 0.8
    stepsize_B1 = stepsize_B2 = stepsize
    t_idx = len(Y.shape) - 1

    for it in range(max_iter):
        A1_T = tl.tenalg.mode_dot(B1, Phi_A1, 2)
        A2_T = tl.tenalg.mode_dot(B2, Phi_A2, 2)
        
        theta_pre = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
        
        if type_outcome == 'Gaussian':
            L_nabula = -Y + theta_pre
        elif type_outcome == 'Binary':
            L_nabula = -Y + expit(theta_pre)
        elif type_outcome == 'Survival_Sarah':
            pi = np.clip(expit(theta_pre), 1e-10, 1 - 1e-10)
            L_nabula = np.zeros_like(theta_pre)
            for n in range(N):
                for d in range(D):
                    t = event_times[n, d]
                    if t < T:
                         # For all times before t
                        L_nabula[n, d, :t] = pi[n, d, :t]
                         # At time t
                        if Y[n,d,t]==1:  # Event occurred
                            L_nabula[n, d, t] = pi[n, d, t]-1
                        else:     # Censored
                            L_nabula[n,d,t] = pi[n,d,t]
                    else: # Right-censored at the end of the study (t == T)
                        L_nabula[n, d, :] = pi[n, d, :]


        L_nabula_mode0 = np.stack([tl.unfold(L_nabula[:, :, idx], mode=0) for idx in range(L_nabula.shape[t_idx])], axis=t_idx)
        L_nabula_mode1 = np.stack([tl.unfold(L_nabula[:, :, idx], mode=1) for idx in range(L_nabula.shape[t_idx])], axis=t_idx)
        
        loss_pre = compute_loss(Y, event_times, theta_pre, type_outcome)
        
        # Update B2
        grad_B2 = tl.tenalg.mode_dot(np.einsum('ijt, jrt -> irt', L_nabula_mode1, A1_T), Phi_A2.T, 2)
        B2_tilde = B2 - stepsize_B2 * grad_B2

        loss_after = compute_loss(Y, event_times, np.einsum("irt, jrt -> ijt", A1_T, tl.tenalg.mode_dot(B2_tilde, Phi_A2, 2)), type_outcome)
 
    
        if loss_after > loss_pre - alpha * stepsize_B2 * 1/np.linalg.norm(grad_B2):
            stepsize_B2 *= beta
        else:
            B2 = B2_tilde
            A2_T = tl.tenalg.mode_dot(B2, Phi_A2, 2)
        
        # Update B1
        grad_B1 = tl.tenalg.mode_dot(np.einsum('ijt, jrt -> irt', L_nabula_mode0, A2_T), Phi_A1.T, 2)
        B1_tilde = B1 - stepsize_B1 * grad_B1
        B1_tilde = tl.tenalg.mode_dot(B1_tilde, X @ np.linalg.pinv(X.T @ X) @ X.T, mode=0)
        loss_after = compute_loss(Y, event_times, np.einsum("irt, jrt -> ijt", tl.tenalg.mode_dot(B1_tilde, Phi_A1, 2), A2_T), type_outcome)
        
        if loss_after > loss_pre - alpha * stepsize_B1 * 1/np.linalg.norm(grad_B1):
            stepsize_B1 *= beta
        else:
            B1 = B1_tilde
            A1_T = tl.tenalg.mode_dot(B1, Phi_A1, 2)
        
        theta_after = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
        loss_after = compute_loss(Y, event_times, theta_after, type_outcome)
        
        tol_temp = np.abs(np.linalg.norm(loss_pre) - np.linalg.norm(loss_after)) / np.linalg.norm(loss_pre)
        
        if not (it % 100):
            print(f'{it}th iteration with loss: {np.round(loss_after,3)}')
        
        if tol_temp < tol and (stepsize_B1 < 1e-6 or stepsize_B2 < 1e-6):
            print('Stop: not enough improvement')
            break
            
    return B1, B2, loss_after

def compute_loss(Y, event_times, theta, type_outcome):
    N = Y.shape[0]
    if type_outcome == 'Gaussian':
        return np.sum((Y - theta)**2) / N
    elif type_outcome == 'Binary':
        return (- Y * theta + np.log(1 + np.exp(theta))).sum() / N
    elif type_outcome == 'Survival_Sarah':
        return survival_likelihood(Y, event_times, theta)
    else:
        raise ValueError("Unsupported outcome type")
    


import numpy as np
import tensorly as tl
from scipy.sparse.linalg import svds
from tensorly.decomposition import parafac

def initialize_tensor_model(Y, K, X, Phi_T):
    N, D, T = Y.shape

    # CP decomposition
    factors = parafac(Y, rank=K)
    theta_CP = tl.cp_to_tensor(factors)

    # SVD-based initialization
    A1, w, A2 = svds(Y.mean(axis=2), k=K)  # average over time
    A1 = A1 @ np.diag(np.sqrt(w))
    A2 = (np.diag(np.sqrt(w)) @ A2).T

    Phi_A1 = Phi_A2 = Phi_T

    # Project A1 onto the space spanned by the time bases
    B1 = tl.tenalg.mode_dot(np.repeat(A1[:, :, np.newaxis], T, axis=2),
                            np.linalg.pinv(Phi_A1.T @ Phi_A1) @ Phi_A1.T, mode=2)
    
    # Project B1 onto the space spanned by the covariates
    B1 = tl.tenalg.mode_dot(B1, X @ np.linalg.pinv(X.T @ X) @ X.T, mode=0)

    # Project A2 onto the space spanned by the time bases
    B2 = tl.tenalg.mode_dot(np.repeat(A2[:, :, np.newaxis], T, axis=2),
                            np.linalg.pinv(Phi_A2.T @ Phi_A2) @ Phi_A2.T, mode=2)

    return B1, B2, theta_CP
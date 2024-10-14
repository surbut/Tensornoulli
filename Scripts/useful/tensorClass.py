import numpy as np
import tensorly as tl
from scipy.special import expit, legendre
from scipy.sparse.linalg import svds
from tensorly.decomposition import parafac
from utils_binaryTensor_Sarah import survival_likelihood

class TensorOptimizer:
    def __init__(self, Y, event_times, X, K, type_outcome, max_iter=500, tol=1e-5):
        self.Y = Y
        self.event_times = event_times
        self.X = X
        self.K = K
        self.type_outcome = type_outcome
        self.max_iter = max_iter
        self.tol = tol
        self.N, self.D, self.T = Y.shape
        self.initialize_parameters()

    def generate_polynomial_bases(self, T, K):
        T_vec = np.linspace(0, 1, T)
        Phi = np.zeros((T, K))
        for k in range(K):
            Phi[:, k] = legendre(k)(2 * T_vec - 1)
        Phi, _ = np.linalg.qr(Phi)
        return Phi

    def create_theta(self):
        A2_T = tl.tenalg.mode_dot(self.B2, self.Phi_A2, 2)  
        A1_T = tl.tenalg.mode_dot(self.B1, self.Phi_A1, 2)  
        theta_fit = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
        return theta_fit, A1_T, A2_T

    def initialize_parameters(self):
        self.Phi_T = self.generate_polynomial_bases(self.T, self.K)
        self.B1, self.B2, _ = self.initialize_tensor_model()
        self.Phi_A1 = self.Phi_A2 = self.Phi_T

    def initialize_tensor_model(self):
        # SVD-based initialization
        A1, w, A2 = svds(self.Y.mean(axis=2), k=self.K)  # average over time
        A1 = A1 @ np.diag(np.sqrt(w))
        A2 = (np.diag(np.sqrt(w)) @ A2).T

        # Project A1 onto the space spanned by the time bases
        B1 = tl.tenalg.mode_dot(np.repeat(A1[:, :, np.newaxis], self.T, axis=2),
                                np.linalg.pinv(self.Phi_T.T @ self.Phi_T) @ self.Phi_T.T, mode=2)
        
        # Project B1 onto the space spanned by the covariates
        B1 = tl.tenalg.mode_dot(B1, self.X @ np.linalg.pinv(self.X.T @ self.X) @ self.X.T, mode=0)

        # Project A2 onto the space spanned by the time bases
        B2 = tl.tenalg.mode_dot(np.repeat(A2[:, :, np.newaxis], self.T, axis=2),
                                np.linalg.pinv(self.Phi_T.T @ self.Phi_T) @ self.Phi_T.T, mode=2)

        return B1, B2, None

    def optimize(self):
        stepsize = 0.1
        alpha = 0.5
        beta = 0.8
        stepsize_B1 = stepsize_B2 = stepsize
        t_idx = len(self.Y.shape) - 1

        for it in range(self.max_iter):
            A1_T = tl.tenalg.mode_dot(self.B1, self.Phi_A1, 2)
            A2_T = tl.tenalg.mode_dot(self.B2, self.Phi_A2, 2)
            
            theta_pre = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
            
            L_nabula = self.compute_L_nabula(theta_pre)

            L_nabula_mode0 = np.stack([tl.unfold(L_nabula[:, :, idx], mode=0) for idx in range(L_nabula.shape[t_idx])], axis=t_idx)
            L_nabula_mode1 = np.stack([tl.unfold(L_nabula[:, :, idx], mode=1) for idx in range(L_nabula.shape[t_idx])], axis=t_idx)
            
            loss_pre = self.compute_loss(theta_pre)
            
            # Update B2
            grad_B2 = tl.tenalg.mode_dot(np.einsum('ijt, jrt -> irt', L_nabula_mode1, A1_T), self.Phi_A2.T, 2)
            B2_tilde = self.B2 - stepsize_B2 * grad_B2

            loss_after = self.compute_loss(np.einsum("irt, jrt -> ijt", A1_T, tl.tenalg.mode_dot(B2_tilde, self.Phi_A2, 2)))
     
            if loss_after > loss_pre - alpha * stepsize_B2 * 1/np.linalg.norm(grad_B2):
                stepsize_B2 *= beta
            else:
                self.B2 = B2_tilde
                A2_T = tl.tenalg.mode_dot(self.B2, self.Phi_A2, 2)
            
            # Update B1
            grad_B1 = tl.tenalg.mode_dot(np.einsum('ijt, jrt -> irt', L_nabula_mode0, A2_T), self.Phi_A1.T, 2)
            B1_tilde = self.B1 - stepsize_B1 * grad_B1
            B1_tilde = tl.tenalg.mode_dot(B1_tilde, self.X @ np.linalg.pinv(self.X.T @ self.X) @ self.X.T, mode=0)
            loss_after = self.compute_loss(np.einsum("irt, jrt -> ijt", tl.tenalg.mode_dot(B1_tilde, self.Phi_A1, 2), A2_T))
            
            if loss_after > loss_pre - alpha * stepsize_B1 * 1/np.linalg.norm(grad_B1):
                stepsize_B1 *= beta
            else:
                self.B1 = B1_tilde
                A1_T = tl.tenalg.mode_dot(self.B1, self.Phi_A1, 2)
            
            theta_after = np.einsum("irt, jrt -> ijt", A1_T, A2_T)
            loss_after = self.compute_loss(theta_after)
            
            tol_temp = np.abs(np.linalg.norm(loss_pre) - np.linalg.norm(loss_after)) / np.linalg.norm(loss_pre)
            
            if not (it % 100):
                print(f'{it}th iteration with loss: {np.round(loss_after,3)}')
            
            if tol_temp < self.tol and (stepsize_B1 < 1e-6 or stepsize_B2 < 1e-6):
                print('Stop: not enough improvement')
                break
                
        return self.B1, self.B2, loss_after

    def compute_L_nabula(self, theta):
        if self.type_outcome == 'Gaussian':
            return -self.Y + theta
        elif self.type_outcome == 'Binary':
            return -self.Y + expit(theta)
        elif self.type_outcome == 'Survival_Sarah':
            pi = np.clip(expit(theta), 1e-10, 1 - 1e-10)
            L_nabula = np.zeros_like(theta)
            for n in range(self.N):
                for d in range(self.D):
                    t = self.event_times[n, d]
                    if t < self.T:
                        L_nabula[n, d, :t] = pi[n, d, :t]
                        if self.Y[n,d,t] == 1:  # Event occurred
                            L_nabula[n, d, t] = pi[n, d, t] - 1
                        else:  # Censored
                            L_nabula[n,d,t] = pi[n,d,t]
                    else:  # Right-censored at the end of the study (t == T)
                        L_nabula[n, d, :] = pi[n, d, :]
            return L_nabula
        else:
            raise ValueError("Unsupported outcome type")

    def compute_loss(self, theta):
        if self.type_outcome == 'Gaussian':
            return np.sum((self.Y - theta)**2) / self.N
        elif self.type_outcome == 'Binary':
            return (- self.Y * theta + np.log(1 + np.exp(theta))).sum() / self.N
        elif self.type_outcome == 'Survival_Sarah':
            return survival_likelihood(self.Y, self.event_times, theta)
        else:
            raise ValueError("Unsupported outcome type")
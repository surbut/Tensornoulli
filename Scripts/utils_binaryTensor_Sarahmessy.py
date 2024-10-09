
from scipy.special import expit
import numpy as np
import tensorly as tl
from scipy.sparse.linalg import svds    
import matplotlib.pyplot as plt 
import numpy as np  
from scipy.special import expit, logit      
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        'U2': self.U2,
        'W': self.W,
        'U3': self.U3,
        'G': self.G,
        'B': self.B,
        'C': self.C
    }
    

    def compute_U1G(self):
        # let's start with no genetic effect
        genetic_effect = np.einsum('np,pkr->nkr', self.G, self.B)
        U1G = genetic_effect #+ self.C
        return U1G

    def compute_theta(self):
        # recall that lambda_k is U1(G) ⊗1 U2 and phi_k is W ⊗2 U3
        if not self.initialized:
            raise ValueError("Parameters have not been initialized yet.")
        U1G = self.compute_U1G()
        lambda_k = np.einsum('nkr,tr->nkt', U1G, self.U2)
        phi = np.einsum('dkr,tr->dkt', self.W, self.U3)
        theta = np.einsum('nkt,dkt->ndt', lambda_k, phi)
        return theta

    def generate_data(self=2):
        """
        Generate survival data based on the model's current parameters.
        """
        theta_true = self.compute_theta()
        pi_true = expit(theta_true)
        return generate_survival_data(pi_true, self.N, self.D, self.T,late_bias=2)
    
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
        # recall that d_theta is the gradient of the log likelihood with respect to theta
        d_lambda = np.einsum('ndt,dkt->nkt', d_theta, phi)
        d_phi = np.einsum('ndt,nkt->dkt', d_theta, lambda_k)
        
        d_U1G = np.einsum('nkt,tr->nkr', d_lambda, self.U2)
        d_W = np.einsum('dkt,tr->dkr', d_phi, self.U3)
        
        d_B = np.einsum('nkr,np->pkr', d_U1G, self.G)
        d_C = d_U1G
        
        return d_W, d_B, d_C

    def fit(self, Y, S, num_iterations=1000, learning_rate=1e-4, l2_reg=1e-5, clip_value=1.0, patience=50):
        losses = []
        best_loss = -np.inf
        patience_counter = 0
        
        for iteration in range(num_iterations):
            log_likelihood = self.survival_likelihood(Y, S)
            loss = log_likelihood - l2_reg * (np.sum(self.W**2) + np.sum(self.B**2) + np.sum(self.C**2))
            losses.append(loss)
            
            if loss > best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iteration}")
                    break
            
            d_W, d_B, d_C = self.compute_gradients(Y, S)
            
            d_W = np.clip(d_W, -clip_value, clip_value)
            d_B = np.clip(d_B, -clip_value, clip_value)
            d_C = np.clip(d_C, -clip_value, clip_value)
            
            self.W += learning_rate * (d_W - 2 * l2_reg * self.W)
            self.B += learning_rate * (d_B - 2 * l2_reg * self.B)
            self.C += learning_rate * (d_C - 2 * l2_reg * self.C)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Log-likelihood: {log_likelihood}")
        
        return losses

#######


    ## beta 'ish', bernstein polynomials
def create_smooth_basis(T, R):
    t = np.linspace(0, 1, T)
    basis = np.zeros((T, R))
    basis[:, 0] = 4 * (1 - t)**3  # early peaking
    basis[:, 1] = 27 * t * (1 - t)**2  # middle peaking
    basis[:, 2] = 4 * t**3  # late peaking
    return basis
    




def generate_survival_data(pi_true, N, D, T):
    """
    Generate survival data based on the true probabilities from the tensor model.
    
    Parameters:
    pi_true (numpy.ndarray): True probabilities from the tensor model, shape (N, D, T)
    N (int): Number of individuals
    D (int): Number of diseases
    T (int): Number of time points
    
    Returns:
    Y (numpy.ndarray): Binary event indicator, shape (N, D, T)
    S (numpy.ndarray): Time of event or censoring, shape (N, D)
    """
    Y = np.zeros((N, D, T), dtype=int)
    S = np.full((N, D), T - 1)  # Initialize all to last time point (censored)

    for n in range(N):
        for d in range(D):
            for t in range(T):
                # Generate a Bernoulli random variable for each time point
                if np.random.random() < pi_true[n, d, t]:
                    Y[n, d, t] = 1
                    S[n, d] = t
                    break  # Stop at the first occurrence of the event
    
    return Y, S
# Set random seed for reproducibility

def compute_gene_effect(model, gene_index):
    effect = np.einsum('np,pkr,tr->nt', model.G[:, gene_index:gene_index+1], 
                       model.B[gene_index:gene_index+1, :, :], model.U2)
    return effect

def plot_genetic_effects_over_time(model, num_individuals=3, num_genes=2):
    plt.figure(figsize=(12, 8))
    for n in range(num_individuals):
        effect = np.einsum('p,pkr,tr->t', model.G[n, :], model.B[:, :, :], model.U2)
        plt.plot(range(model.T), effect, label=f'Individual {n+1}')
    plt.title('Genetic Effects Over Time')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Effect')
    plt.show()

def plot_gene_heatmap(model, gene_index, num_individuals=20):
    gene_effect = compute_gene_effect(model, gene_index)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(gene_effect[:num_individuals], 
                cmap='coolwarm', center=0, 
                xticklabels=range(0, model.T, model.T//10),
                yticklabels=range(1, num_individuals+1))
    plt.title(f'Genetic Effect of Gene {gene_index+1} Across Individuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Individual')
    plt.show()

def plot_average_gene_effects(model, num_genes=5):
    plt.figure(figsize=(12, 8))
    for gene_index in range(num_genes):
        gene_effect = compute_gene_effect(model, gene_index)
        average_effect = gene_effect.mean(axis=0)
        plt.plot(range(model.T), average_effect, label=f'Gene {gene_index+1}')

    plt.title('Average Genetic Effect Over Time')
    plt.xlabel('Time')
    plt.ylabel('Average Effect')
    plt.legend()
    plt.show()

def plot_basis_functions(model):
    plt.figure(figsize=(10, 5))
    for r in range(model.R1):
        plt.plot(range(model.T), model.U2[:, r], label=f'Basis {r+1}')
    plt.title('Individual Temporal Basis Functions')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Basis Function Value')
    plt.show()

def plot_disease_probabilities(model, pi, num_individuals=3, num_diseases=3):
    plt.figure(figsize=(12, 8))
    for n in range(num_individuals):
        for d in range(num_diseases):
            plt.plot(range(model.T), pi[n, d, :], label=f'Individual {n+1}, Disease {d+1}')
    plt.title('Disease Probability Over Time')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.show()

def plot_survival_times(S, T):
    plt.figure(figsize=(10, 5))
    plt.hist(S.flatten(), bins=T, range=(0, T-1), align='left')
    plt.title('Distribution of Survival Times')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()





from scipy.special import expit
import numpy as np
import tensorly as tl
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from utils_binaryTensor_Sarah import (TensorModelWithGenetics, generate_survival_data,plot_genetic_effects_over_time, plot_gene_heatmap,
                                      plot_average_gene_effects, plot_basis_functions,
                                      plot_disease_probabilities, plot_survival_times)



N, P, D, K, R1, R2, T = 1000, 20, 5, 4, 3, 3, 50

# Generate synthetic data
true_model = TensorModelWithGenetics(N, P, D, K, R1, R2, T)
true_model.initialize_parameters()

theta_true = true_model.compute_theta()
pi_true = expit(theta_true)

G = true_model.G # the genetic covariates for each individual
U2 = true_model.U2
U3 = true_model.U3
W=true_model.W
U1G = true_model.compute_U1G()
# Compute lambda_k and phi
lambda_k = tl.tenalg.mode_dot(U1G, true_model.U2, 2)
phi = tl.tenalg.mode_dot(true_model.W, true_model.U3, 2)
# Compute theta
Y, S = generate_survival_data(pi_true, N, D, T)
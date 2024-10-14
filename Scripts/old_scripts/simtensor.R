

# TensorNoulli Model Simulation in R

# Set random seed for reproducibility
set.seed(42)

# Define dimensions
N <- 1000  # number of individuals
D <- 20    # number of diseases
K <- 4    # number of topics
R <- 3     # number of temporal bases (early, middle, late)
T <- 50    # number of time points

# Create smooth temporal bases (U2 and U3) using polynomials
create_smooth_basis <- function(T, R) {
  t <- seq(0, 1, length.out = T)
  basis <- array(0, dim = c(T, R))
  basis[, 1] <- 4 * (1 - t)^3  # early peaking
  basis[, 2] <- 27 * t * (1 - t)^2  # middle peaking
  basis[, 3] <- 4 * t^3  # late peaking
  return(basis)
}

U2 <- create_smooth_basis(T, R)
U3 <- create_smooth_basis(T, R)

matplot(U2, type = "l", xlab = "Time", ylab = "Basis Value", main = "Temporal Basis U2 for disease progression through any topic ")

matplot(U3, type = "l", xlab = "Time", ylab = "Basis Value", main = "Temporal Basis U3 for individual progression through any topic ")

# Create disease weights (W, which
W <- array(runif(D * K * R, -10, -8), dim = c(D, K, R))
for (d in 1:D) {
  for (k in 1:K) {
    peak_time <- sample(1:R, 1)
    W[d, k, peak_time] <- runif(1, -5, -4)  # Make one time period dominant
  }
}


# Create individual loadings (U1)
U1 <- array(0, dim = c(N, K, R))
for (i in 1:N) {
  for(k in 1:K){
    peak_time <- sample(1:R, 1)
    U1[i, k, peak_time] <- runif(1, 0.001, 0.002)  # Make one time period dominant
  }
}


# individual time stuff

# Create individual time weights (U3)
A = array(0, dim = c(N, K, T))
for (i in 1:N) {
  for (k in 1:K) {
    A[i, k, ] <- U1[i, k, ] %*% t(U2)
  }
}



matplot(t(A[sample(N,1), , ]), type = "l", xlab = "Time", ylab = "Individual Time Weight", main = "Individual Time Weight for Individual 1 and Topic 1")


# Create individual time weights (U3)
B = array(0, dim = c(D, K, T))
for (d in 1:D) {
  for (k in 1:K) {
    B[d, k, ] <- W[d, k, ] %*% t(U3)
  }
}



matplot(t(B[sample(D,1), , ]), type = "l", xlab = "Time", ylab = "Disease Time Weight", main = "Disease Time Weight for Individual 1 and Topic 1")


# Compute theta
compute_theta <- function(U1, U2, W, U3) {
  theta <- array(0, dim = c(N, T, D))
  for (i in 1:N) {
    for (d in 1:D) {
      for (t in 1:T) {
        for (k in 1:K) {
          for (r in 1:R) {
            theta[i, t, d] <- theta[i, t, d] + U1[i, k, r] * U2[t, r] * W[d, k, r] * U3[t, r]
          }
        }
      }
    }
  }
  return(theta)
}

theta <- compute_theta(U1, U2, W, U3)


plot_topic_all_diseases <- function(W, U3,topic) {

  disease_time=W[, topic, ] %*% t(U3)
  matplot(t(disease_time), type = "l", xlab = "Time", ylab = "Probability", main = paste0("All Diseases in Topic",topic))

}

phi = array(0, dim = c(dim(W)[2], D, T))
for (k in 1:K) {
  phi[k, , ] = W[, k, ] %*% t(U3)
}


 sample_disease=sample(D,3)
  par(mfrow = c(1, K), mar = c(4, 4, 2, 1))
  for (k in 1:K)
   matplot(t(phi[k, , ]), type = "l", xlab = "Time", ylab = "Probability", main = paste0("Topic",k))

 sample_disease=sample(D,3)
  par(mfrow = c(1, length(sample_disease)), mar = c(4, 4, 2, 1))
  for (d in sample_disease)
   matplot(t(phi[, d, ]), type = "l", xlab = "Time", ylab = "Probability", main = paste0("Disease",d))






lambda = array(0, dim = c(N,dim(U1)[2], T))

  for(k in 1:K){
  lambda[,k, ] = U1[,k,] %*% t(U2)
}
# 4. Plot topic progression across topics for a few sample individuals

matplot(t(lambda[sample(N,1),,]), type = "l", xlab = "Time", ylab = "Probability", main = "Individual 1 in all topics")


matplot(t(lambda[,1,]), type = "l", xlab = "Time", ylab = "Probability", main = "Topic 1 in all People")

matplot(theta[1,,])

par(mfrow=c(1,3))
image((theta[sample(N,1),,]), main = "Individual 1",xlab="Time",ylab="Disease")


image((theta[sample(N,1),,]), main = "Individual 2",xlab="Time",ylab="Disease")


image(plogis(theta[sample(N,1),,]), main = "Individual 3",xlab="Time",ylab="Disease")




y = array(0, dim = c(N, D, T))

# Simulate data
for (i in 1:N) {
  for (d in 1:D) {
    for (t in 1:T) {
      if (sum(y[i, d, 1:t]) == 0) {
        # Disease hasn't occurred yet

        # Simulate diagnosis
        y[i, d, t] <- rbinom(1, 1, plogis(theta[i, t, d])/10)
      } else {
        break  # Stop once disease is diagnosed
      }
    }
  }
}

image(y[1,,])



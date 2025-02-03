from numpy.random import normal,uniform, multivariate_normal
import numpy as np
def generate_joint_gaussian_random_variables(sample_size,n_var):
    # u = normal(0,1,(n_var,sample_size))
    # v = normal(0,1,(n_var,sample_size))
    # a = uniform(2,(n_var))
    # b = uniform(2,(n_var))
    # c = uniform(2,(n_var))
    # d = uniform(2,(n_var))
    # X = a * u + b * v
    # Y = c * u + d * v
    # mean = [0]*n_var
    # cov = [[1, 0.5], [0.5, 1]]

    # A = np.random.rand(n_var, n_var)
    # B = np.dot(A, A.transpose())

    # samples = multivariate_normal(mean=mean, cov=B,size =(sample_size,5))
    # print(samples)
    # # exit()
    # return samples[:,0],samples[:,1]

    # Define dimensions
    d = 5
    m = 5

    mu_X = np.array([0, 0, 0, 0, 0])
    mu_Y = np.array([0, 0, 0, 0, 0])

    mu = np.concatenate([mu_X, mu_Y])

    # Sigma_XX = np.array([[1.0, 0.5, 0.3],
    #                     [0.5, 1.0, 0.2],
    #                     [0.3, 0.2, 1.0]])

    # Sigma_YY = np.array([[1.5, 0.7, 0.4],
    #                     [0.7, 1.2, 0.6],
    #                     [0.4, 0.6, 1.8]])

    # Sigma_XY = np.array([[0.4, 0.2, 0.1],
    #                     [0.3, 0.5, 0.2],
    #                     [0.6, 0.8, 0.3]])

    # Sigma_YX = Sigma_XY.T

    A = np.random.uniform( 0,2, (d + m, d + m) )   # Random matrix
    Sigma = A @ A.T
    # print(Sigma)
    # exit()
    # _,Sigma = np.linalg.eig((A @ A.T))  # Ensure it's symmetric and positive semi-definite

    # Generate multivariate normal samples
    data = np.random.multivariate_normal(mu, Sigma, sample_size)
    # Sigma = np.block([[Sigma_XX, Sigma_XY], 
    #                 [Sigma_YX, Sigma_YY]])
    # data = np.random.multivariate_normal(mu, Sigma, sample_size)
    X = data[:, :d].T
    Y = data[:, d:].T
    return X,Y,Sigma


if __name__ == "__main__":
    generate_joint_gaussian_random_variables(100,10)
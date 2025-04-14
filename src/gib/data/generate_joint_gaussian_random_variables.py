from numpy.random import uniform, multivariate_normal
from numpy import zeros

def generate_joint_gaussian_random_variables(sample_size:int, n_var:int ):
    """
    generate random jointe gaussian random variable
    return sample data and covariance matrix, µ is 0
    """

    d = n_var[0]
    m = n_var[1]
    mu = zeros(d+m)
    A = uniform( 0,1, (d + m,d + m))
    Sigma = A @ A.T

    data = multivariate_normal(mu, Sigma, sample_size)
    X = data[:, :d].T
    Y = data[:, d:].T
    return X,Y,Sigma
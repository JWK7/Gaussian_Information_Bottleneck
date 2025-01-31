from numpy.random import normal,uniform
def generate_joint_gaussian_random_variables(sample_size,n_var):
    u = normal(0,1,(n_var,sample_size))
    v = normal(0,1,(n_var,sample_size))
    a = uniform(10,(n_var))
    b = uniform(10,(n_var))
    c = uniform(10,(n_var))
    d = uniform(10,(n_var))
    X = a * u + b * v
    Y = c * u + d * v
    return X,Y


if __name__ == "__main__":
    generate_joint_gaussian_random_variables(100)
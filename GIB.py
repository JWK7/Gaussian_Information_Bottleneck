from generate_joint_gaussian_random_variables import generate_joint_gaussian_random_variables
import numpy as np

def get_epsilon(data_size , cov):
    return np.random.normal(0,cov,data_size)

# def direct_A(**kwargs):
#     return np.linalg.eig(np.matmul(kwargs["cov_XgY"], np.linalg.inv(kwargs["cov_X"])))


def get_covs(A,B):
    cov = np.cov(A,B)
    cov_AA = cov[:cov.shape[0]//2,:cov.shape[0]//2]
    cov_AB = cov[:cov.shape[0]//2,cov.shape[0]//2:]
    cov_BB = cov[cov.shape[0]//2:,cov.shape[0]//2:]
    cov_AgB = cov_AA - np.matmul( np.matmul(cov_AB , np.linalg.inv(cov_BB)) , cov_AB.T )
    return cov_AA,cov_AB,cov_BB,cov_AgB

def get_alpha(**kwargs):
    r = get_r(**kwargs)
    return ( ( kwargs["Beta"]*(1-kwargs["lamda"])-1)  / (r*kwargs["lamda"]) )

def get_critical_points(lamda): return 1/(1-lamda)

def get_init_A(**kwargs):
    #V is full size A while D is the diagonal matrix of lambda
    cov_X,_,_,cov_XgY = get_covs(kwargs["X"],kwargs["Y"])
    lamda,V= np.linalg.eig(np.matmul(cov_XgY, np.linalg.inv(cov_X)))
    alpha = get_alpha(lamda=lamda,cov_X=cov_X,V=V,Beta = kwargs["Beta"])
    B_c = get_critical_points(lamda)
    full_A = alpha*V
    return ( alpha * (kwargs["Beta"] < B_c) * full_A )

def get_init_Cov_Epsil(A_shape):
    return np.cov(np.random.normal(-0.01,0.01,A_shape))

def get_r(**kwargs):
    return np.diagonal( np.matmul( np.matmul(kwargs["V"].T,kwargs["cov_X"]) , kwargs["V"]) )

def get_T(A,X,cov_epsil):
    # print(np.random.multivariate_normal([0,0,0,0,0],cov_epsil).shape)
    # print(np.matmul(A,X).shape)[0,]*A.shape[0]
    AX = np.matmul(A,X)
    epsil = np.random.multivariate_normal([0,]*A.shape[0],cov_epsil)
    return np.add(epsil.reshape(*epsil.shape,1),AX)

def get_p(X,Y):
    I = np.identity(X.shape[0])
    _,_,_,covYgX = get_covs(Y,X)
    covX = np.cov(X)
    return I - np.matmul(covYgX,covX)

def GIB(X,Y):
    Beta = 1.005
    A = get_init_A(X=X,Y=Y,Beta=Beta)
    print(A)
    cov_epsil = get_init_Cov_Epsil(A.shape)

    #p = I - cov(Y|X) inv(cov(X)) 
    p = get_p(X,Y)
    if (np.allclose(A, 0)): raise SystemExit('B < B_c1')

    for i in range(100):
        T = get_T(A,X,cov_epsil)
        cov_T, _,_,cov_TgY = get_covs(T,Y)
        cov_epsil = np.linalg.inv(Beta * np.linalg.inv(cov_TgY) - (Beta-1) * np.linalg.inv(cov_T))
        A = Beta * np.matmul( np.matmul( np.matmul(cov_epsil,np.linalg.inv(cov_TgY)) , A) , p)
        # print(i)
    print(A)
        

def main():
    X,Y = generate_joint_gaussian_random_variables(20,5)
    GIB(X,Y)

if __name__ == "__main__":
    main()
from generate_joint_gaussian_random_variables import generate_joint_gaussian_random_variables
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

def get_epsilon(data_size , cov):
    return np.random.multivariate_normal([0]*cov.shape[0],cov,data_size)


def get_covs(cov):
    cov_AA = cov[:cov.shape[0]//2,:cov.shape[0]//2]
    cov_AB = cov[:cov.shape[0]//2,cov.shape[0]//2:]
    cov_BB = cov[cov.shape[0]//2:,cov.shape[0]//2:]
    cov_AgB = cov_AA - cov_AB  @  np.linalg.inv(cov_BB) @ cov_AB.T
    return cov_AA,cov_AB,cov_BB,cov_AgB

def get_alpha(**kwargs):
    r = get_r(**kwargs)
    return ( ( kwargs["Beta"]*(1-kwargs["lamda"])-1)  / (r*kwargs["lamda"]) )

def get_critical_points(lamda): return 1/(1-lamda)

def get_init_A(**kwargs):
    #V is full size A while D is the diagonal matrix of lambda
    cov_X,_,_,cov_XgY = get_covs(kwargs["Sigma"])
    lamda ,_ = np.linalg.eig(cov_XgY @ np.linalg.inv(cov_X))
    _, V = np.linalg.eig((cov_XgY @ np.linalg.inv(cov_X)).T)
    V = (np.linalg.inv(V))
    # lamda,V,_ = eig(cov_XgY @ np.linalg.inv(cov_X),left=True)

    sort_index = np.argsort(lamda)
    lamda = lamda[sort_index]
    V = V[:,sort_index]

    alpha = get_alpha(lamda=lamda,cov_X=cov_X,V=V,Beta = kwargs["Beta"])
    B_c = get_critical_points(lamda)
    if all(x == False for x in (kwargs["Beta"] > B_c)): print('B < B_c1: Full Noise')
    if all(x == True for x in (kwargs["Beta"] > B_c)): print('B > B_cn: No Compression')
    # if all(x == False for x in (kwargs["Beta"] > B_c)): raise SystemExit('B < B_c1: Full Noise')
    # if all(x == True for x in (kwargs["Beta"] > B_c)): raise SystemExit('B > B_cn: No Compression')
    full_A = np.expand_dims(alpha,-1)*V
    return np.expand_dims( (kwargs["Beta"] > B_c),-1)* full_A

def get_init_Cov_Epsil(A_shape):
    return np.identity(A_shape[0])*0.3
    E = np.random.uniform( 0,1, A_shape)
    # _,E = np.linalg.eig(E @ E.T)
    return E@E.T

def get_r(**kwargs):
    return np.diagonal( kwargs["V"].T@kwargs["cov_X"] @kwargs["V"])

def get_T(A,X,cov_epsil):
    AX = A@X
    epsil = np.random.multivariate_normal([0,]*A.shape[0],cov_epsil,X.shape[1]).T
    # for i in range(X.shape[1]):
        # X[:,i] = np.random.multivariate_normal([0,]*A.shape[0],cov_epsil)
    return epsil + AX

def get_p(Sigma):
    _,_,covX,covYgX = get_covs(Sigma[::-1,::-1])
    I = np.identity(covX.shape[0])
    return I - covYgX @ np.linalg.inv(covX)

def GIB(B,Sigma):
    A = get_init_A(Beta=B,Sigma=Sigma)
    cov_epsil = get_init_Cov_Epsil(A.shape)
    # cov_epsil = np.identity(A.shape[0])

    #p = I - cov(Y|X) inv(cov(X)) 
    p = get_p(Sigma)
    cov_X,cov_XY,cov_Y,cov_XgY = get_covs(Sigma)

    cov_T = A@cov_X@A.T + cov_epsil
    cov_TX = A@cov_X 
    cov_TY = A@cov_XY
    
    IXT = get_mutual_information(cov_X,cov_TX.T,cov_T)
    ITY = get_mutual_information(cov_T,cov_TY,cov_Y)
    return IXT,ITY

def get_entropy(d,Sigma):
    # print(np.log(np.linalg.det(Sigma)))
    return (d/2)*np.log(1+2*np.pi) + (1/2)*np.log(np.linalg.det(Sigma))

def get_mutual_information(cov_A,cov_AB,cov_B):
    Sigma = np.zeros( ( 2*cov_A.shape[0] , 2*cov_A.shape[0] ) )
    Sigma[:Sigma.shape[0]//2,:Sigma.shape[0]//2] = cov_A
    Sigma[:Sigma.shape[0]//2,Sigma.shape[0]//2:] = cov_AB
    Sigma[Sigma.shape[0]//2:,:Sigma.shape[0]//2] = cov_AB.T
    Sigma[Sigma.shape[0]//2:,Sigma.shape[0]//2:] = cov_B
    # print(np.linalg.det(Sigma))
    # print(cov_A.shape)
    # print(np.linalg.det(Sigma))
    H_A = get_entropy(cov_A.shape[0],cov_A)
    H_B = ( get_entropy(cov_B.shape[0],cov_B))
    H = ( get_entropy(Sigma.shape[0],Sigma))
    return (1/2)*np.log(H_A*H_B/H)

def Iterative_GIB(X,Y,Sigma):

    Beta = 1.1
    A = get_init_A(Beta=Beta,Sigma=Sigma)
    cov_epsil = get_init_Cov_Epsil(A.shape)
    # cov_epsil = np.identity(A.shape[0])

    #p = I - cov(Y|X) inv(cov(X)) 
    p = get_p(Sigma)
    cov_x,_,_,cov_xgy = get_covs(Sigma)

    # print(A) 
    # exit()


    for i in range(10000):
        # T = get_T(A,X,cov_epsil)
        # cov_T, _,_,cov_TgY = get_covs( np.cov(T,Y) )
        # print(cov_T)
        cov_T = A@cov_x@A.T+ cov_epsil
        # print(cov_xgy)
        # print(A)
        # print(A@ cov_xgy @A.T+ cov_epsil)
        # exit()
        cov_TgY = A@ cov_xgy @A.T + cov_epsil
        # input()
        # exit()
        # print(cov_epsil)
        # print(np.cov(T,Y))
        # print()
        # print(cov_T)
        # print(cov_epsil)
        # print(cov_TgY)
        # exit()
        print(cov_epsil)
        cov_epsil = np.linalg.inv(Beta * np.linalg.inv(cov_TgY) - (Beta-1) * np.linalg.inv(cov_T))
        A_prev = A
        A = Beta * cov_epsil @ np.linalg.inv(cov_TgY) @ A @  p
        # print(A)
        print( (A-A_prev)[0][0] )

        # if i == 0:
        #     exit()
        # print(A)
        # print(cov_epsil)
        # exit()
        # print(A)
    print(A)
def PlotOverB():
    _,_,Sigma = generate_joint_gaussian_random_variables(10000,5)
    cov_X,cov_XY,covY,covXgY = get_covs(Sigma)
    eigenval,_ = np.linalg.eig(covXgY@np.linalg.inv(cov_X))
    eigenval = (np.sort(eigenval))
    Bs = 1/(1-eigenval)
    Bs = np.append( Bs, (Bs[-1]+1) )
    # print(range(1000))
    # exit()
    IXTs = []
    ITYs = []
    # for B in Bs:
    for B in range(6000):
        # print(B/10)
        IXT, ITY = GIB(B/10,Sigma)
        IXTs.append(IXT)
        ITYs.append(ITY)
    plt.plot(IXTs,ITYs)
    plt.savefig("GIB_Information_Curve.png")
    

def main():
    PlotOverB()
    # X,Y,Sigma = generate_joint_gaussian_random_variables(100000,5)
    # Iterative_GIB(X,Y,Sigma)

if __name__ == "__main__":
    main()
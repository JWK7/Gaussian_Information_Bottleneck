import numpy as np
from scipy.linalg import eig

class GIB:
    def __init__(self, **kwargs)-> None:
        self.Sigma = kwargs["Sigma"]
        if "iterations" in kwargs:
            self.iterations = kwargs["iterations"]
        self.mode = kwargs['mode']
        self.n_var = kwargs["n_var"]
        self.init_covs()
        self.init_eig()
        self.init_r()
        self.init_critical_points()
        self.init_p()
        self.init_Cov_Epsil(kwargs["n_var"][0],kwargs["c"])

    #p = I - Σy|x @ inv(Σx)
    def init_p(self)-> None: self.p = np.identity(self.cov_X.shape[0]) - self.cov_XgY @ np.linalg.inv(self.cov_X)

    def init_r(self)-> None: self.r = np.diagonal( self.V@self.cov_X@self.V.T)

    def init_critical_points(self)-> None: self.B_c = 1/(1-self.lamda)

    def init_eig(self)-> None:
        lamda,V = eig(self.cov_XgY @ np.linalg.inv(self.cov_X),left=True,right=False)
        lamda= np.real(lamda)
        V = np.real(V).T
        sort_index = np.argsort(lamda)
        self.lamda = lamda[sort_index]
        self.V = V[sort_index,:]

    def init_covs(self)-> None:
        self.cov_X = self.Sigma[:self.Sigma.shape[0]//2,:self.Sigma.shape[0]//2]
        self.cov_XY = self.Sigma[:self.Sigma.shape[0]//2,self.Sigma.shape[0]//2:]
        self.cov_Y = self.Sigma[self.Sigma.shape[0]//2:,self.Sigma.shape[0]//2:]
        self.cov_XgY = self.cov_X - self.cov_XY  @  np.linalg.inv(self.cov_Y) @ self.cov_XY.T
        self.cov_YgX = self.cov_Y - self.cov_XY.T  @  np.linalg.inv(self.cov_X) @ self.cov_XY
        return

    def init_Cov_Epsil(self,d:int, c:int = 1)-> None: self.cov_epsil = np.identity(d) * c

    def get_GIB_type(self)-> None:
        if self.mode == "direct":
            return self.direct_GIB
        elif self.mode == "iterative":
            return  self.iterative_GIB
        elif self.mode == "gradient":
            return self.gradient_GIB
        else:
            print("Undefined Mode")
            exit()

    def run(self,beta:int, num_V:tuple = None)-> None:
        GIB_type = self.get_GIB_type()
        return GIB_type(beta,num_V)

    def update_cov_Ts(self)-> None:
        self.cov_T = self.A@self.cov_X@self.A.T + self.cov_epsil
        self.cov_TX = self.A@self.cov_X
        self.cov_TY = self.A@self.cov_XY
        self.cov_TgY = self.A@self.cov_XgY@self.A.T + self.cov_epsil

    def algebraic_A(self,beta:int , num_V:tuple = None):
        self.beta = beta
        self.alpha = self.get_alpha()
        A = np.expand_dims(self.alpha,-1)*self.V
        if num_V != None:
            A[num_V:,:] = 0
            self.A = A
            return
        if all(x == False for x in (beta > self.B_c)): print('B < B_c1: Full Noise')
        if all(x == True for x in (beta > self.B_c)): print('B > B_cn: No Compression')
        self.A = A
        return
    
    def iterative_GIB(self,beta:int , num_V:int = None)-> None:
        self.A = np.identity(self.X.shape[0])
        self.cov_epsil = np.identity(self.X.shape[0])
        self.beta = beta
        for i in range(self.iterations):
            self.update_cov_Ts()
            self.cov_epsil = np.linalg.inv( self.beta * np.linalg.inv(self.cov_TgY) - (self.beta-1) * np.linalg.inv(self.cov_T) ) 
            self.A = self.beta * self.cov_epsil @ np.linalg.inv(self.cov_TgY) @ self.A @ self.p
        IXT,ITY = self.get_gaussian_mutual_information()
        return IXT,ITY

    def gradient_GIB(self,beta:int, num_V:int = None)-> None:
        self.A = np.identity(self.n_var[0])
        self.cov_epsil = np.identity(self.n_var[0])
        self.beta = beta
        for i in range(self.iterations):
            self.A -= 0.1 * self.dL()
        IXT,ITY = self.get_gaussian_mutual_information()
        return IXT,ITY

    def dL(self)-> None:
        idx = ((1-self.beta)
               * np.linalg.inv( self.A@self.cov_X@self.A.T + np.identity(self.cov_epsil.shape[0]) )
               @ (2*self.A@self.cov_X))
        idy = (self.beta 
               * np.linalg.inv(self.A @ self.cov_XgY @ self.A.T + np.identity(self.cov_epsil.shape[0])) 
               @ (2*self.A@self.cov_XgY))
        return 1/2 * (idx + idy)

    def direct_GIB(self,beta:int ,num_V:int = None)-> None:
        self.algebraic_A(beta = beta,num_V = num_V)
        self.update_cov_Ts()
        IXT,ITY = self.get_gaussian_mutual_information()
        return IXT,ITY

    def get_gaussian_mutual_information(self)-> None:
        IXT = np.log(np.linalg.det(self.A@self.cov_X@self.A.T+self.cov_epsil)) - np.log(np.linalg.det(self.cov_epsil)) 
        ITY = np.log(np.linalg.det(self.A@self.cov_X@self.A.T+self.cov_epsil)) - np.log(np.linalg.det(self.A@self.cov_XgY@self.A.T+self.cov_epsil))
        return IXT,ITY

    def L(self)-> None: return (1-self.beta) * np.log(np.linalg.det(self.A@self.cov_X@self.A.T+self.cov_epsil)) - np.log(np.linalg.det(self.cov_epsil)) + self.beta*np.log(np.linalg.det(self.A@self.cov_XgY@self.A.T+self.cov_epsil))

    def get_alpha(self)-> None:
        idx =  ( ( self.beta*(1-self.lamda)-1)  / (self.lamda * self.r) )
        idx[idx<0] = 0
        return np.sqrt(idx)
from plot import plot_beta_theta,plot_information_curve
from gib import gib
from gib.data.generate_joint_gaussian_random_variables import generate_joint_gaussian_random_variables
import sys
from scipy.linalg import eig

def b_vs_theta():
    calc_thetas =[]
    Bs = []
    for i in range(0,30000):
        B = 10**(i/10000)
        calc_thetas.append(1-1/B)
        Bs.append(B)
    plot_beta_theta(Bs,calc_thetas)

def information_curve(
        GIB, scale = 0.1, B_range = [0,10],
        step = 0.1):
    IXTss = []
    ITYss = []
    for num_V in range(len(GIB.B_c)):
        IXTs = []
        ITYs = []
        for B in range(*[int(element / step) for element in B_range]):
            IXT, ITY = GIB.run(beta=0.1+scale*B,num_V=num_V+1)
            IXTs.append(IXT)
            ITYs.append(ITY)
        IXTss.append(IXTs)
        ITYss.append(ITYs)
    plot_information_curve(IXTss,ITYss)

def compare_modes(
        **kwargs):
    IXTss = []
    ITYss = []
    for mode in kwargs["modes"]:
        print(mode)
        IXTs = []
        ITYs = []
        GIB = gib.GIB(Sigma = kwargs["Sigma"], n_var = kwargs["n_var"], mode = mode , c= 1, iterations=10000)
        for B in range(1000):
            print(B)
            IXT, ITY = GIB.run(beta=0.1+0.1*B)
            IXTs.append(IXT)
            ITYs.append(ITY)
        IXTss.append(IXTs)
        ITYss.append(ITYs)

        # IXTs = []
        # ITYs = []
        # IXT, ITY = GIB.run(beta= 1.2)
    plot_information_curve(IXTss,ITYss,kwargs["modes"])


    pass

def main(sample_size,n_var):
    X,Y,Sigma = generate_joint_gaussian_random_variables(sample_size,n_var)
    compare_modes(Sigma=Sigma,sample_size=sample_size,n_var=n_var,modes=["direct","gradient"])

    return


if __name__ == "__main__":
    main(10,(4,4))
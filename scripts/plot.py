import matplotlib.pyplot as plt

def plot_information_curve(IXTss: list,ITYss: list,labels: list)-> None:
    if labels == None:
        labels = range(len(IXTss))

    for i in range(len(IXTss)):
        plt.plot(IXTss[i],ITYss[i],label=labels[i])
        print(i)
    plt.legend()
    plt.xlabel("I[X;T]")
    plt.ylabel("I[T;Y]")
    plt.savefig("results/GIB_Information_Curve.png")
    plt.cla()

def plot_beta_theta(Bs: list,calc_thetas: list)-> None:
    plt.plot(Bs,calc_thetas)
    plt.xscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\theta$')
    plt.savefig("results/BetaVsTheta.png")
    plt.cla()
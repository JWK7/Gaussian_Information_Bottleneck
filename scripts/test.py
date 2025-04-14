from gib.gib import GIB
import sys
import numpy as np

def b_vs_theta():
    GIB_class = test_class()
    pass

def test_class(sample_size = 100000 ,n_var = (4,4), mode = "direct" , c= 1):
    GIB_class = GIB(sample_size = sample_size ,n_var = n_var, mode = mode , c = c)

def main():

    np.array([[ 0.51933613 ,-1.36532957,  1.15868738 , 3.06531984],
              [ 0.3464161 ,  0.90920334 ,-0.78828878,  0.11122853],
              [ 0.    ,     -0.      ,   -0.      ,    0.  ,      ],
              [-0.     ,     0.        , -0.       ,   0.  ,      ]])
    M = np.array([[ 0.24741953, -0.04370617 , 0.03152584 , 0.81575872],
              [ 0.08278728 , 1.00021631, -0.86000461, -0.80518907],
              [-0.05730314 ,-0.91829882,  0.78911939 , 0.79740214],
              [ 0.56423428 ,-0.9193258 ,  0.77501254 , 2.73110154]])
    while(1):
        print(M)
        target = int(input("Row to Affect:"))-1
        row_from = int(input("Row from:"))-1
        scalar = int(input("index:"))-1
        M[target,:] -=  (M[target,scalar]/M[row_from,scalar]) *  M[row_from,:]
    
if __name__ == "__main__":
    main()
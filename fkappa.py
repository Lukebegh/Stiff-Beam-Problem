import numpy as np
from compute_v import compute_v
from compute_C import compute_C
from compute_D import compute_D

def fkappa(t,theta,kappa,S):
    C=compute_C(theta,S)
    D=compute_D(theta,S)
    v=compute_v(t,theta,S)
    #set up some of the terms we will need
    C_inv=np.linalg.inv(C)
    DC_inv=np.matmul(D,C_inv)
    w=np.matmul(D,v)+np.square(kappa)
    #now bringing the terms together into the desired equation of motion
    eom=np.matmul(C,v)+np.matmul(DC_inv,w)

    return eom
import numpy as np

def compute_C(theta,S):
    #initialise
    Cfunc=np.zeros((S,S))
    #boundary values
    Cfunc[0,0]=1
    Cfunc[S-1,S-1]=3
    #diagonal entries
    for i in range(S-2):
        Cfunc[i+1,i+1]=2
    #other entries above and below the diagonal
    for i in range(S-1):
        Cfunc[i,i+1]=-np.cos(theta[i]-theta[i+1])
        Cfunc[i+1,i]=-np.cos(theta[i]-theta[i+1])

    return Cfunc
import numpy as np

def compute_D(theta,S):
    #initialise
    Dfunc=np.zeros((S,S))
    #entries above and below the diagonal
    for i in range(S-1):
        Dfunc[i,i+1]=-np.sin(theta[i]-theta[i+1])
        Dfunc[i+1,i]=-np.sin(theta[i+1]-theta[i])

    return Dfunc
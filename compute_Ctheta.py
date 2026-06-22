import numpy as np

#Computes the partial derivatives of C with respect to theta
#computation done by hand. This function only creates the tensor.
def compute_Ctheta(theta,S):
    #index 1 is component of theta that we differentiate by
    Ctheta=np.zeros((S,S,S))
    #enter the boundary values
    Ctheta[0,0,1]=np.sin(theta[0]-theta[1])
    Ctheta[0,1,0]=-np.sin(theta[1]-theta[0])
    Ctheta[S-1,S-2,S-1]=-np.sin(theta[S-2]-theta[S-1])
    Ctheta[S-1,S-1,S-2]=np.sin(theta[S-1]-theta[S-2])
    #loop over the components of theta
    for i in range(S-2):
        Ctheta[i+1,i,i+1]=-np.sin(theta[i]-theta[i+1])
        Ctheta[i+1,i+1,i]=np.sin(theta[i+1]-theta[i])
        Ctheta[i+1,i+1,i+2]=np.sin(theta[i+1]-theta[i+2])
        Ctheta[i+1,i+2,i+1]=-np.sin(theta[i+2]-theta[i+1])

    return Ctheta

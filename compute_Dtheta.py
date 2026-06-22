import numpy as np

#Computes the partial derivatives of D with respect to theta
#computation done by hand. This function only creates the tensor.
def compute_Dtheta(theta,S):
    #index 1 is component of theta that we differentiate by
    Dtheta=np.zeros((S,S,S))
    #enter the boundary values
    Dtheta[0,0,1]=-np.cos(theta[0]-theta[1])
    Dtheta[0,1,0]=np.cos(theta[1]-theta[0])
    Dtheta[S-1,S-2,S-1]=np.cos(theta[S-2]-theta[S-1])
    Dtheta[S-1,S-1,S-2]=-np.cos(theta[S-1]-theta[S-2])
    #loop over the components of theta
    for i in range(S-2):
        Dtheta[i+1,i,i+1]=np.cos(theta[i]-theta[i+1])
        Dtheta[i+1,i+1,i]=-np.cos(theta[i+1]-theta[i])
        Dtheta[i+1,i+1,i+2]=-np.cos(theta[i+1]-theta[i+2])
        Dtheta[i+1,i+2,i+1]=np.cos(theta[i+2]-theta[i+1])

    return Dtheta
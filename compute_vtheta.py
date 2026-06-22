import numpy as np
import math

#Computes the partial derivatives of v with respect to theta
def compute_vtheta(t,theta,S):
    #set up Gtilda as in solutions for finding the Jacobian
    Gtilda=np.zeros((S,S))
    Gtilda[0,0]=-3
    Gtilda[S-1,S-1]=-1
    for i in range(S-2):
        Gtilda[i+1,i+1]=-2
    for i in range(S-1):
        Gtilda[i+1,i]=1
        Gtilda[i,i+1]=1
    #form diagonal matrix making up second term of v'
    vterm2=np.zeros((S,S))
    for i in range(S):
        vterm2[i,i]=np.cos(theta[i])-np.sin(theta[i])
    #compute v'
    if t<math.pi:
        vtheta=(S**4)*Gtilda+(S**2)*1.5*((np.sin(t))**2)*vterm2
    else:
        vtheta=(S**4)*Gtilda

    return vtheta
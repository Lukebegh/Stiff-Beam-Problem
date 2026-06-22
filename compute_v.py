import numpy as np
import math

def compute_v(t,theta,S):
    #Case when force is applied t in [0,pi]
    if t<math.pi:
        #initialise
        vfunc=np.zeros(S)
        #boundary values l=1,S
        vfunc[0]=(S**4)*(-3*theta[0]+theta[1])+(S**2)*1.5*((np.sin(t))**2)*(np.cos(theta[0])+np.sin(theta[0]))
        vfunc[S-1]=(S**4)*(theta[S-2]-theta[S-1])+(S**2)*1.5*((np.sin(t))**2)*(np.cos(theta[S-1])+np.sin(theta[S-1]))
        #l=2,...,S-1
        for i in range(S-2):
            vfunc[i+1]=(S**4)*(theta[i]-2*theta[i+1]+theta[i+2])+(S**2)*1.5*((np.sin(t))**2)*(np.cos(theta[i+1])+np.sin(theta[i+1]))
    #Case when no force applied t>=pi
    else:
        #initialise
        vfunc=np.zeros(S)
        #boundary values l=1,S
        vfunc[0]=(S**4)*(-3*theta[0]+theta[1])
        vfunc[S-1]=(S**4)*(theta[S-2]-theta[S-1])
        #l=2,...,S-1
        for i in range(S-2):
            vfunc[i+1]=(S**4)*(theta[i]-2*theta[i+1]+theta[i+2])

    return vfunc
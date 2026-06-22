import numpy as np
from compute_v import compute_v
from compute_C import compute_C
from compute_D import compute_D
from compute_vtheta import compute_vtheta
from compute_Ctheta import compute_Ctheta
from compute_Dtheta import compute_Dtheta

#This is a function to compute the Jacobian of the thetas
def compute_Jtheta(t,theta,kappa,S):
    #prepare Jtheta, a matrix of the partial deriv of the ith component of f by the jth component of theta
    Jtheta=np.zeros((S,S))
    #bring in required matrices and vectors
    C=compute_C(theta,S)
    D=compute_D(theta,S)
    v=compute_v(t,theta,S)
    vtheta=compute_vtheta(t,theta,S)
    Ctheta=compute_Ctheta(theta,S)
    Dtheta=compute_Dtheta(theta,S)
    #compute term 1 in my equation (written on paper)
    term1=np.zeros((S,S))
    for j in range(S):
        term1[:,j]=np.matmul(Ctheta[j,:,:],v)
    #term2
    term2=np.zeros((S,S))
    for j in range(S):
        term2[:,j]=np.matmul(C,vtheta[:,j])
    #term3
    w=np.matmul(D,v)+np.square(kappa)
    C_inv=np.linalg.inv(C)
    u=np.matmul(C_inv,w)
    term3=np.zeros((S,S))
    for j in range(S):
        term3[:,j]=np.matmul(Dtheta[j,:,:],u)
    #term 4
    term4=np.zeros((S,S))
    for j in range(S):
        term4[:,j]=np.matmul(Ctheta[j,:,:],u)
        term4[:,j]=np.matmul(C_inv,term4[:,j])
        term4[:,j]=-np.matmul(D,term4[:,j])
    #term5
    term5=np.zeros((S,S))
    for j in range(S):
        term5[:,j]=np.matmul(Dtheta[j,:,:],v)
        term5[:,j]=np.matmul(C_inv,term5[:,j])
        term5[:,j]=np.matmul(D,term5[:,j])
    #term6
    term6=np.zeros((S,S))
    for j in range(S):
        term6[:,j]=np.matmul(D,vtheta[:,j])
        term6[:,j]=np.matmul(C_inv,term6[:,j])
        term6[:,j]=np.matmul(D,term6[:,j])

    #Now sum to obtain Jtheta
    Jtheta=term1+term2+term3+term4+term5+term6

    return Jtheta

# This is a function to compute the Jacobian of the kappas
def compute_Jkappa(t, theta, kappa, S):
    # prepare Jkappa, a matrix of the partial deriv of the ith component of f by the jth component of kappa
    Jkappa = np.zeros((S, S))
    #bring in required matrices and vectors
    C=compute_C(theta,S)
    D=compute_D(theta,S)
    v=compute_v(t,theta,S)
    vtheta=compute_vtheta(t,theta,S)
    Ctheta=compute_Ctheta(theta,S)
    Dtheta=compute_Dtheta(theta,S)

    #prepare diagkappa and use express in given solutions for the Jacobian
    diagkappa=np.zeros((S, S))
    for j in range(S):
        diagkappa[j,j]=kappa[j]
    #compute Jkappa
    C_inv = np.linalg.inv(C)
    Jkappa=np.matmul(C_inv,diagkappa)
    Jkappa=2*np.matmul(D,Jkappa)

    return Jkappa
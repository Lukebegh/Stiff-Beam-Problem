import numpy as np
import matplotlib.pyplot as plt
from fkappa import fkappa
from compute_J import compute_Jtheta
from compute_J import compute_Jkappa

#IMPORTANT: For instructions on how to use this code, please skip past the stiffbeam function

#
##
### START OF OUR FUNCTION ###

def stiffbeam(S,T,L,m,method,f):

    #
    ##
    ### SETUP OF THE PROBLEM ###

    #time step size h
    h=T/m
    #time vector t
    t=np.zeros(m)
    for i in range(m):
        t[i]=i*h
    #Set the method to use based on the value inputted for the indicator
    if method==1:
        #Explicit Euler
        methodname="Explicit Euler"
        alpha=np.array([0])
        gamma=np.array([1])
        Beta=np.array([0])
        plicit=0
    elif method==2:
        #Modified Euler
        methodname="Modified Euler"
        alpha=np.array([0,0.5])
        gamma=np.array([0,1])
        Beta=np.array([[0,0],[0.5,0]])
        plicit=0
    elif method==3:
        #Heun
        methodname="Heun"
        alpha=np.array([0,1])
        gamma=np.array([0.5,0.5])
        Beta=np.array([[0,0],[1,0]])
        plicit=0
    elif method==4:
        #Classical RK4
        methodname= "Classical RK4"
        alpha=np.array([0,0.5,0.5,1])
        gamma=np.array([1/6,1/3,1/3,1/6])
        Beta=np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
        plicit=0
    elif method==5:
        #Hammer and Hollingsworth (FP iteration)
        methodname="Hammer&Hollingsworth with FP iteration"
        p=(3**0.5)/6
        alpha=np.array([0.5-p,0.5+p])
        gamma=np.array([0.5,0.5])
        Beta=np.array([[0.25,0.25-p],[0.25+p,0.25]])
        plicit=1
    elif method==6:
        #Hammer and Hollingsworth (Newton iteration)
        methodname= "Hammer&Hollingsworth with Newton iteration"
        p=(3**0.5)/6
        alpha=np.array([0.5-p,0.5+p])
        gamma=np.array([0.5,0.5])
        Beta=np.array([[0.25,0.25-p],[0.25+p,0.25]])
        plicit=2
    elif method==0:
        #This one is for a customised method. Please fill in the template below as desired (enter the Butcher Tableau)
        methodname="customized"
        alpha=np.array([0,0])
        gamma=np.array([0,0])
        Beta=np.array([[0,0],[0,0]])
        #Please choose whether your custom method is explicit (0), implicit with FP iteration (1) or implicit with Newton iteration (2)
        plicit=0

    #Compute number of stages s of the method
    s=np.size(alpha)
    #Turn RKM method into Nystroem method
    #if statement ensures the correct format for future matrix multiplication
    if s==1:
        gammabar=np.array([np.matmul(np.transpose(gamma),Beta)])
        Betabar=np.array([np.matmul(Beta,Beta)])
    else:
        gammabar=np.matmul(np.transpose(gamma),Beta)
        Betabar=np.matmul(Beta,Beta)
    #set up theta matrix of angles. each column will represent a point on the beam, each row a point in time.
    theta=np.zeros((m,S))
    #define kappa, the rate of change of the thetas, similarly
    kappa=np.zeros((m,S))

    ### END OF SETUP ###
    ##
    #

    #
    ##
    ### PROCEDURE ###
    # Compute kdash as in Nystroem methods (we showed in lectures that we can ignore k itself)
    # Follow the Nystroem method for each time step, each time obtaining a vector of the thetas and a vector of the kappas
    # Use these to compute the new kdash for the next step.
    # If method is implicit, use iteration to find the kdash.

    #Explicit method case
    if plicit==0:
        for l in range(m-1):
            #kdash will be a matrix containing information about the s stages in the rows, and each k is a vector of dimension S (denoted in the columns)
            #we need to recalculate this at each time step
            kdash=np.zeros((s,S))
            for i in range(s):
                for j in range(S):
                    #creating matrices that I will need to use in the expression for kdash
                    #if statement ensures the correct format for future matrix multiplication
                    if s==1:
                        Betabar_kdash=np.array([np.matmul(Betabar,kdash)])
                        Beta_kdash=np.array([np.matmul(Beta,kdash)])
                    else:
                        Betabar_kdash=np.matmul(Betabar,kdash)
                        Beta_kdash=np.matmul(Beta,kdash)
                    #fkappa is the discretised equation of motion, ie. the rate of change of kappa. It is a vector of dimension S
                    kdash[i,j]=fkappa(t[l]+alpha[i]*h,theta[l,:]+alpha[i]*h*kappa[l,:]+(h**2)*Betabar_kdash[i,:],kappa[l,:]+h*Beta_kdash[i,:],S)[j]

            theta[l+1,:]=theta[l,:]+h*kappa[l,:]+(h**2)*np.array([np.matmul(np.transpose(gammabar),kdash)])
            kappa[l+1,:]=kappa[l,:]+h*np.array([np.matmul(np.transpose(gamma),kdash)])


    #'Implicit method with FP' case
    elif plicit==1:
        for l in range(m-1):
            #kdash will be a matrix as in the explicit case, but with an extra dimension to denote the iteration
            kdash=np.zeros((f,s,S))
            #iterate over index a. I have used a parallel inner iteration, starting with 0 at each time step
            for a in range(f-1):
                for i in range(s):
                    for j in range(S):
                        #creating matrices that I will need to use in the expression for kdash
                        #if statement ensures the correct format for future matrix multiplication
                        if s==1:
                            Betabar_kdash=np.array([np.matmul(Betabar,kdash[a,:,:])])
                            Beta_kdash=np.array([np.matmul(Beta,kdash[a,:,:])])
                        else:
                            Betabar_kdash=np.matmul(Betabar,kdash[a,:,:])
                            Beta_kdash=np.matmul(Beta,kdash[a,:,:])
                        #Fixed point iteration
                        kdash[a+1,i,j]=fkappa(t[l]+alpha[i]*h,theta[l,:]+alpha[i]*h*kappa[l,:]+(h**2)*Betabar_kdash[i,:],kappa[l,:]+h*Beta_kdash[i,:],S)[j]


            theta[l+1,:]=theta[l,:]+h*kappa[l,:]+(h**2)*np.array([np.matmul(np.transpose(gammabar),kdash[f-1,:,:])])
            kappa[l+1,:]=kappa[l,:]+h*np.array([np.matmul(np.transpose(gamma),kdash[f-1,:,:])])


    #'Implicit method with Newton' case
    elif plicit==2:
        #Construct 4d identity for use in the Newton iteration
        Id=np.zeros((s,S,s,S))
        for i in range(s):
            for j in range(S):
                for k in range(s):
                    for l in range(S):
                        if i==k:
                            if j==l:
                                Id[i,j,k,l]=1
        #loop over time steps
        for l in range(m-1):
            #kdash will be a matrix as in the explicit case, but with an extra dimension to denote the iteration
            kdash=np.zeros((f,s,S))
            #iterate over index a
            for a in range(f-1):
                for j in range(S):
                    #creating matrices that I will need to use in the expression for kdash
                    #if statement ensures the correct format for future matrix multiplication
                    if s==1:
                        Betabar_kdash=np.array([np.matmul(Betabar,kdash[a,:,:])])
                        Beta_kdash=np.array([np.matmul(Beta,kdash[a,:,:])])
                    else:
                        Betabar_kdash=np.matmul(Betabar,kdash[a,:,:])
                        Beta_kdash=np.matmul(Beta,kdash[a,:,:])
                    #Create some matrices for use in iteration
                    #define function F such that k'=F(k')
                    F=np.zeros((s,S))
                    for b in range(s):
                        for c in range(S):
                            F[b,c]=fkappa(t[l]+alpha[b]*h,theta[l,:]+alpha[b]*h*kappa[l,:]+(h**2)*Betabar_kdash[b,:],kappa[l,:]+h*Beta_kdash[b,:],S)[c]
                    #Build Jacobian tensor DF of F with respect to k' (this is a matrix function of a matrix)
                    DF=np.zeros((s,S,s,S))
                    for b in range(s):
                        Jtheta=compute_Jtheta(t[l]+alpha[b]*h,theta[l,:]+alpha[b]*h*kappa[l,:]+(h**2)*Betabar_kdash[b,:],kappa[l,:]+h*Beta_kdash[b,:],S)
                        Jkappa=compute_Jkappa(t[l]+alpha[b]*h,theta[l,:]+alpha[b]*h*kappa[l,:]+(h**2)*Betabar_kdash[b,:],kappa[l,:]+h*Beta_kdash[b,:],S)
                        for k in range(s):
                            for d in range(S):
                                for q in range(S):
                                    DF[b,d,k,q]=(h**2)*Betabar[b,k]*Jtheta[d,q]+h*Beta[b,k]*Jkappa[d,q]
                    #Newton iteration (I derived a Matrix Newton iteration formula from principles)
                    kdash[a+1,:,:]=kdash[a,:,:]-np.linalg.tensorsolve(Id-DF,kdash[a,:,:]-F[:,:])


            theta[l+1,:]=theta[l,:]+h*kappa[l,:]+(h**2)*np.array([np.matmul(np.transpose(gammabar),kdash[f-1,:,:])])
            kappa[l+1,:]=kappa[l,:]+h*np.array([np.matmul(np.transpose(gamma),kdash[f-1,:,:])])


    ### END OF PROCEDURE ###
    ##
    #

    #
    ##
    ### GRAPHING THE RESULTS ###
    #

    #Convert our thetas to cartesian (x,y) coordinates for graphing
    #x
    x=np.zeros((m,S+2))
    for l in range(m):
        x[l,1]=0.5*L/S
        for i in range(S-1):
            x[l,i+2]=x[l,i+1]+np.cos(theta[l,i])*L/S
        x[l,S+1]=x[l,S]+np.cos(theta[l,S-1])*0.5*L/S

    #y
    y=np.zeros((m,S+2))
    for l in range(m):
        for i in range(S-1):
            y[l,i+2]=y[l,i+1]+np.sin(theta[l,i])*L/S
        y[l,S+1]=y[l,S]+np.sin(theta[l,S-1])*0.5*L/S

    #plotting
    #we will plot a maximum of 30 beams
    if m<=30:
        m_plot=m
        time_plot=np.zeros(m)
        indices=np.zeros(m)
        for i in range(m):
            time_plot[i]=(i*5)/m
            indices[i]=i
    else:
        m_plot =30
        #discretization of the time interval in 30 time steps:
        time_plot=np.zeros(30)
        time_method=np.zeros(m)
        indices=np.zeros(30)
        for j in range(m):
            time_method[j]=(j*5)/m
        for i in range(30):
            idx = np.argmin(np.abs(time_method - (i*5)/30))  # Index of the number of old discretization with minimal distance to new discretization
            time_plot[i]=time_method[idx]
            indices[i]=idx

    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Time t')
    ax.set_ylabel('x-Axis')
    ax.set_zlabel('y-Axis')
    ax.set_title(f"Graph of beam replicated by the {methodname} Method using {m} time steps.")
        
    #loop over time steps to extract one graph of the beam at a time (for every time step)
    for n in range(m_plot):
        time=np.zeros(S+2)
        xcoord=np.zeros(S+2)
        ycoord=np.zeros(S+2)
        #loop over every point on the beam to extract the values needed
        for i in range(S+2):
            time[i]=time_plot[n]
            xcoord[i]=x[int(indices[n]),i]
            ycoord[i]=y[int(indices[n]),i]
        #plot this one beam-graph
        ax.plot(time,ycoord, xcoord, color='blue', linewidth=0.5)

    plt.show()


    ### END OF GRAPHING ###
    ##
    #

### END OF FUNCTION ###
##
#

##################################  HOW TO USE THE CODE  ########################################
#There are 3 options:	-Use the demo code which displays the cases shown in the lecture slides one by one
#			-Customise the inputs in the INPUT section below, and pick one of the RKMs offered
#			-Customise the inputs in the INPUT section below, and customise the RKM used by
#			selecting method=0 and inputting your custom Butcher Tableau using the template
#			in the SETUP section within the function above
#
#
# Please note that the code sometimes errors if the number of time steps or number of iterations selected is not sufficient
# To avoid this I have included some recommended values for which the code should work

#
##
### START OF INPUTS ###

#Select demo=1 to use the demo code, demo=0 to use custom inputs. If using demo code, you may disregard the rest of the input section.
demo=1
#discretisation - number of points on beam (Tested with S=8)
S=8
#time interval we are working on: [0,T)  (Tested with T=5)
T=5
#Beam length L
L=1
#Number of time steps m (Recommended: similar values to those in lectures for the corresponding method used)
m=30000
#Please enter for the 'method' variable the number of the method you wish to use.
#Key:   Explicit Euler: 1
#       Modified Euler: 2
#       Heun: 3
#       Classical RK4: 4
#       Hammer and Hollingsworth (FP iteration): 5
#       Hammer and Hollingsworth (Newton iteration): 6
#       Other method of your choice: Please enter 0 and fill in the details of your method in the 'Setup' section of the code
method=1
#Number of iterations desired f if the method is implicit (recommended 50 for FP and 10 for Newton, I have tested these)
f=10

### END OF INPUTS ###
##
#

if demo==1:
    #
    ##
    ### START OF DEMO-CODE: PUSH RUN IF YOU WANT TO REPLICATE THE EXAMPLES ON THE SLIDES
    #first, the explicit euler method:
    for m in [30000, 40000, 50000]:
        stiffbeam(8,5,1,m,1,10)

    #the modified euler method:
    for m in [2200,2400, 2600]:
        stiffbeam(8,5,1,m,2,10)

    #the Heun's method:
    for m in [2200, 2400, 2600]:
        stiffbeam(8,5,1,m,3,10)

    #the Classical RK4:
    for m in [421,425,430]:
        stiffbeam(8,5,1,m,4,10)

    #the Implicit RK4 with Fixed Point Iteration:
    for m in [360,356]:
        stiffbeam(8,5,1,m,5,50)

    #the Implicit RK4 with Newton Iteration:
    for m in [10,30,50]:
        stiffbeam(8,5,1,m,6,10)

elif demo==0:
    stiffbeam(S,T,L,m,method,f)


#Thank you for taking the time to read, and we wish you the best with using our code
#Best,
#Sonja Eberle, Lucas Beghein



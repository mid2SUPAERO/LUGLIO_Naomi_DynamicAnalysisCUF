# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:06:09 2019

@author: NAOMI
"""

import numpy as np
from numpy import linalg as LA
import pylab as pl
import math as Ma
import scipy.integrate as integ
import scipy.linalg as LAS
from sympy import Symbol
from numpy import matrix
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs,eigsh

"""Datas"""

bnod=3    #bnod is what in the previous sections has been referred as Nn, that is to say the number of nodes for each beam element
nF=9     #nF is what in the previous sections has been referres as M, that is to say the number of terms for each cross section expansion F_\tau
ne_t=1 # number of Lagrange elements that constitute the cross-section
nn_t=9
ne_x=20   #number of longitudinal elements
nn_x=(bnod-1)*ne_x+1  #number of longitudinal nodes
h=1  #edge of the cross-section beam. I suppose a square cross-section
L=20  #longitudinal length of the beam
E=75*(10**9) #Young's Module
tau=0.33  #Poisson ratio
G=E/(2*(1+tau))  #Shear Elastic Constant
rho=2700     #density of material, I have supposed Alluminium
le=L/ne_x    #length of each beam element--> I have used a uniform longitudinal mesh


"""Cx= matrix where the i-th raw has the global index of i-th beam element's longitudinal nodes"""
Cx=[]
for i in range(0,ne_x):
    Cx.append([])
    
    Cx[i].append(i*2)
    Cx[i].append(i*2+1)
    Cx[i].append(i*2+2)
    
# If, for example, bnod was equal to 2-->
#Cx=[]
#for i in range(0,ne_x):
#    Cx.append([])
    
#    Cx[i].append(i)
#    Cx[i].append(i+1)
    

    """Matrix of elasticity for an isotropic material"""
S=np.array([[1/E,-tau/E,-tau/E,0,0,0],[-tau/E,1/E,-tau/E,0,0,0],[-tau/E,-tau/E,1/E,0,0,0],[0,0,0,1/G,0,0],[0,0,0,0,1/G,0],[0,0,0,0,0,1/G]])
C=np.linalg.inv(S)

F=[] #will contain the functions Ftau and its partial derivatives
Kel=[]  #will be the stiffness matrix of a beam element
N=[]  #will contain the interpolatin function along the longitudinal axis

Ctrav=[]  #will be the matrix were the i-th raw refers to the i-th Lagrange element of the cross-section--> the i-th raw will have the global enumaration of that element's nodes
Ctrav.append([1,2,3,4,5,6,7,8,9])

Con=[]  #will be the matrix were each raw will have the y_tau and z_tau real coordinates of the i-th transverse element's nodes
#Con.append([y1,z1,y2,z2,y3,z3,y4,z4])
Con.append([-h/2,-h/2,0,-h/2,h/2,-h/2,h/2,0,h/2,h/2,0,h/2,-h/2,h/2,-h/2,0,0,0])

Nod=[]  #this matrix will contain the normalised coordination of each node of LE9 element
Nod.append([-1,-1])
Nod.append([0,-1])
Nod.append([1,-1])
Nod.append([1,0])
Nod.append([1,1])
Nod.append([0,1])
Nod.append([-1,1])
Nod.append([-1,0])
Nod.append([0,0])

Bar=[]  #The i-th raw contains the baricentre's real coordinates of the i-th Lagrange element
Bar.append([0,0])


def f1t(alpha,beta,alphat,betat):
    return (1/4)*(alpha**2+alpha*alphat)*(beta**2+beta*betat)

#partial derivative of f1 respect to alpha
def df1ta(alpha,beta,alphat,betat):
    return (1/4)*(2*alpha+alphat)*(beta**2+beta*betat)

def df1tb(alpha,beta,alphat,betat):
    return (1/4)*(alpha**2+alpha*alphat)*(2*beta+betat)

def f2t(alpha,beta,alphat,betat):
    return ((1/2)*(betat**2)*(beta**2+beta*betat)*(1-alpha**2)+(1/2)*(alphat**2)*(alpha**2+alpha*alphat)*(1-beta**2))

def df2ta(alpha,beta,alphat,betat):
    return ((1/2)*(betat**2)*(beta**2+beta*betat)*(-2*alpha)+(1/2)*(alphat**2)*(2*alpha+alphat)*(1-beta**2))

def df2tb(alpha,beta,alphat,betat):
    return ((1/2)*(betat**2)*(2*beta+betat)*(1-alpha**2)+(1/2)*(alphat**2)*(alpha**2+alpha*alphat)*(-2*beta))

def f3t(alpha,beta,alphat,betat):
    return (1-alpha**2)*(1-beta**2)

def df3ta(alpha,beta,alphat,betat):
    return (-2*alpha)*(1-beta**2)

def df3tb(alpha,beta,alphat,betat):
    return (1-alpha**2)*(-2*beta)

"""Let's build the nucleus stiffeness matrix"""

"""This function will provide the determinant of the Jacobian matrix linked to the transformation from (y,z)--> (alpha,beta)"""
def Jac(e,alpha,beta):
    F1a=df1ta(alpha,beta,Nod[0][0],Nod[0][1])
    F1b=df1tb(alpha,beta,Nod[0][0],Nod[0][1])
    F2a=df2ta(alpha,beta,Nod[1][0],Nod[1][1])
    F2b=df2tb(alpha,beta,Nod[1][0],Nod[1][1])
    F3a=df1ta(alpha,beta,Nod[2][0],Nod[2][1])
    F3b=df1tb(alpha,beta,Nod[2][0],Nod[2][1])
    F4a=df2ta(alpha,beta,Nod[3][0],Nod[3][1])
    F4b=df2ta(alpha,beta,Nod[3][0],Nod[3][1])
    F5a=df1ta(alpha,beta,Nod[4][0],Nod[4][1])
    F5b=df1tb(alpha,beta,Nod[4][0],Nod[4][1])
    F6a=df2ta(alpha,beta,Nod[5][0],Nod[5][1])
    F6b=df2tb(alpha,beta,Nod[5][0],Nod[5][1])
    F7a=df1ta(alpha,beta,Nod[6][0],Nod[6][1])
    F7b=df1tb(alpha,beta,Nod[6][0],Nod[6][1])
    F8a=df2ta(alpha,beta,Nod[7][0],Nod[7][1])
    F8b=df2ta(alpha,beta,Nod[7][0],Nod[7][1])
    F9a=df3ta(alpha,beta,Nod[8][0],Nod[8][1])
    F9b=df3tb(alpha,beta,Nod[8][0],Nod[8][1])
    
    
    return ((F1a*Con[e][0]+F2a*Con[e][2]+F3a*Con[e][4]+F4a*Con[e][6]+F5a*Con[e][8]+F6a*Con[e][10]+F7a*Con[e][12]+F8a*Con[e][14]+F9a*Con[e][16])*(F1b*Con[e][1]+F2b*Con[e][3]+F3b*Con[e][5]+F4b*Con[e][7]+F5b*Con[e][9]+F6b*Con[e][11]+F7b*Con[e][13]+F8b*Con[e][15]+F9b*Con[e][17])-(F1b*Con[e][0]+F2b*Con[e][2]+F3b*Con[e][4]+F4b*Con[e][6]+F5b*Con[e][8]+F6b*Con[e][10]+F7b*Con[e][12]+F8b*Con[e][14]+F9b*Con[e][16])*(F1a*Con[e][1]+F2a*Con[e][3]+F3a*Con[e][5]+F4a*Con[e][7]+F5a*Con[e][9]+F6a*Con[e][11]+F7a*Con[e][13]+F8a*Con[e][15]+F9a*Con[e][17]))


#This funtion will provide the function to be integrated over the beam element volume
def integrand(beta,alpha,eta,e,c,b,n,a,k,p):
    global F
    F.clear()
    
    
    #Functions Fi
    F1=f1t(alpha,beta,Nod[0][0],Nod[0][1])
    F2=f2t(alpha,beta,Nod[1][0],Nod[1][1])
    F3=f1t(alpha,beta,Nod[2][0],Nod[2][1])
    F4=f2t(alpha,beta,Nod[3][0],Nod[3][1])
    F5=f1t(alpha,beta,Nod[4][0],Nod[4][1])
    F6=f2t(alpha,beta,Nod[5][0],Nod[5][1])
    F7=f1t(alpha,beta,Nod[6][0],Nod[6][1])
    F8=f2t(alpha,beta,Nod[7][0],Nod[7][1])
    F9=f3t(alpha,beta,Nod[8][0],Nod[8][1])
    
    
   
    #Partial derivatives of Fi
    F1a=df1ta(alpha,beta,Nod[0][0],Nod[0][1])
    F1b=df1tb(alpha,beta,Nod[0][0],Nod[0][1])
    F2a=df2ta(alpha,beta,Nod[1][0],Nod[1][1])
    F2b=df2tb(alpha,beta,Nod[1][0],Nod[1][1])
    F3a=df1ta(alpha,beta,Nod[2][0],Nod[2][1])
    F3b=df1tb(alpha,beta,Nod[2][0],Nod[2][1])
    F4a=df2ta(alpha,beta,Nod[3][0],Nod[3][1])
    F4b=df2tb(alpha,beta,Nod[3][0],Nod[3][1])
    F5a=df1ta(alpha,beta,Nod[4][0],Nod[4][1])
    F5b=df1tb(alpha,beta,Nod[4][0],Nod[4][1])
    F6a=df2ta(alpha,beta,Nod[5][0],Nod[5][1])
    F6b=df2tb(alpha,beta,Nod[5][0],Nod[5][1])
    F7a=df1ta(alpha,beta,Nod[6][0],Nod[6][1])
    F7b=df1tb(alpha,beta,Nod[6][0],Nod[6][1])
    F8a=df2ta(alpha,beta,Nod[7][0],Nod[7][1])
    F8b=df2tb(alpha,beta,Nod[7][0],Nod[7][1])
    F9a=df3ta(alpha,beta,Nod[8][0],Nod[8][1])
    F9b=df3tb(alpha,beta,Nod[8][0],Nod[8][1])
    
    #Partial derivatives of y and z respect to alpha and beta
    ya=F1a*Con[e][0]+F2a*Con[e][2]+F3a*Con[e][4]+F4a*Con[e][6]+F5a*Con[e][8]+F6a*Con[e][10]+F7a*Con[e][12]+F8a*Con[e][14]+F9a*Con[e][16]
    yb=F1b*Con[e][0]+F2b*Con[e][2]+F3b*Con[e][4]+F4b*Con[e][6]+F5b*Con[e][8]+F6b*Con[e][10]+F7b*Con[e][12]+F8b*Con[e][14]+F9b*Con[e][16]
    za=F1a*Con[e][1]+F2a*Con[e][3]+F3a*Con[e][5]+F4a*Con[e][7]+F5a*Con[e][9]+F6a*Con[e][11]+F7a*Con[e][13]+F8a*Con[e][15]+F9a*Con[e][17]
    zb=F1b*Con[e][1]+F2b*Con[e][3]+F3b*Con[e][5]+F4b*Con[e][7]+F5b*Con[e][9]+F6b*Con[e][11]+F7b*Con[e][13]+F8b*Con[e][15]+F9b*Con[e][17]
    
    F1y=(1/Jac(e,alpha,beta))*(zb*F1a-za*F1b)
    F1z=(1/Jac(e,alpha,beta))*(-yb*F1a+ya*F1b)
    F2y=(1/Jac(e,alpha,beta))*(zb*F2a-za*F2b)
    F2z=(1/Jac(e,alpha,beta))*(-yb*F2a+ya*F2b)
    F3y=(1/Jac(e,alpha,beta))*(zb*F3a-za*F3b)
    F3z=(1/Jac(e,alpha,beta))*(-yb*F3a+ya*F3b)
    F4y=(1/Jac(e,alpha,beta))*(zb*F4a-za*F4b)
    F4z=(1/Jac(e,alpha,beta))*(-yb*F4a+ya*F4b)
    F5y=(1/Jac(e,alpha,beta))*(zb*F5a-za*F5b)
    F5z=(1/Jac(e,alpha,beta))*(-yb*F5a+ya*F5b)
    F6y=(1/Jac(e,alpha,beta))*(zb*F6a-za*F6b)
    F6z=(1/Jac(e,alpha,beta))*(-yb*F6a+ya*F6b)
    F7y=(1/Jac(e,alpha,beta))*(zb*F7a-za*F7b)
    F7z=(1/Jac(e,alpha,beta))*(-yb*F7a+ya*F7b)
    F8y=(1/Jac(e,alpha,beta))*(zb*F8a-za*F8b)
    F8z=(1/Jac(e,alpha,beta))*(-yb*F8a+ya*F8b)
    F9y=(1/Jac(e,alpha,beta))*(zb*F9a-za*F9b)
    F9z=(1/Jac(e,alpha,beta))*(-yb*F9a+ya*F9b)
    
    F.append(F1)
    F.append(F1y)
    F.append(F1z)
    F.append(F2)
    F.append(F2y)
    F.append(F2z)
    F.append(F3)
    F.append(F3y)
    F.append(F3z)
    F.append(F4)
    F.append(F4y)
    F.append(F4z)
    F.append(F5)
    F.append(F5y)
    F.append(F5z)
    F.append(F6)
    F.append(F6y)
    F.append(F6z)
    F.append(F7)
    F.append(F7y)
    F.append(F7z)
    F.append(F8)
    F.append(F8y)
    F.append(F8z)
    F.append(F9)
    F.append(F9y)
    F.append(F9z)
    
    
    N.clear()

#    if bnod=2--> decomment the following 4 raws
#    N.append((1/2)*(1-(x/(le/2))))
#    N.append(-1/(le))    
#    N.append((1/2)*(1+(x/(le/2))))    
#    N.append(1/(le))  

#    if bnod=3--> decomment the following 6 raws    
    
    N.append((1/2)*eta*(eta-1))
    N.append((1/le)*(2*eta-1))
    
    N.append(+(1+eta)*(1-eta))
    N.append(-(4/le)*eta)
    
    
    N.append((1/2)*eta*(eta+1))
    N.append((1/le)*(2*eta+1))

#    if bnod=4--> decomment the following 8 raws   
#    
#    N.append((-9/16)*(eta+1/3)*(eta-1/3)*(eta-1))
#    N.append((-9*2/(16*le))*((1)*(eta-1/3)*(eta-1)+(eta+1/3)*(1)*(eta-1)+(eta+1/3)*(eta-1/3)*(1)))
#    
#    N.append((27/16)*(eta+1)*(eta-1/3)*(eta-1))
#    N.append((27*2/(16*le))*((1)*(eta-1/3)*(eta-1)+(eta+1)*(1)*(eta-1)+(eta+1)*(eta-1/3)*(1)))
#    
#    N.append((-27/16)*(eta+1)*(eta+1/3)*(eta-1))
#    N.append((-27*2/(16*le))*((1)*(eta+1/3)*(eta-1)+(eta+1)*(1)*(eta-1)+(eta+1)*(eta+1/3)*(1)))
#    
#    N.append((9/16)*(eta+1/3)*(eta-1/3)*(eta+1))
#    N.append((9*2/(16*le))*((1)*(eta-1/3)*(eta+1)+(eta+1/3)*(1)*(eta+1)+(eta+1/3)*(eta-1/3)*(1)))
#    
##        
        
    return N[c]*F[b]*C[n][a]*F[k]*N[p]*Jac(e,alpha,beta)*le/2


#this function will perform the integral of the function integrand over the beam element
#For Lagrange expansions I have performed the Gauss-Legendre Integration
#In this case 3 Points of Gauss have been used

def integral(e,c,b,n,a,k,p):
    xi=[0,-0.774596669241483,+0.7745966692411483]
    w=[0.888888888888889,0.555555555555556,0.555555555555556]
    I=0
    for i in range(0,len(w)):
        for j in range(0,len(w)):
            for m in range(0,len(w)):
                I=I+(integrand(xi[i],xi[j],xi[m],e,c,b,n,a,k,p))*(w[i])*(w[j])*(w[m])
    return I

def integraltot():
    global Kel
    Kel=np.zeros((ne_t*bnod*nF*3,bnod*nF*3)) #if the cross-section has been discretised with multiple Lagrange elements, the matrix Kel will be composed of the sequence of all the stiffeness Lagrange element matrices
    for e in range(0,ne_t):
        for q in range(0,bnod):   #loop over the nodes of a longitudinal element
            for o in range(0,bnod):   #loop over the nodes of a longitudinal element
                for w in range(0,nF):   #loop over the terms of transverse expansion over the cross-section
                    for m in range(0,nF):   #loop over the terms of transverse expansion over the cross-section
                        Kel[e*bnod*nF*3+q*nF*3+w*3][o*nF*3+m*3]=integral(e,q*2+1,w*3,0,0,m*3,o*2+1)+integral(e,q*2,w*3+1,5,5,m*3+1,o*2)+integral(e,q*2,w*3+2,3,3,m*3+2,o*2)
                        Kel[e*bnod*nF*3+q*nF*3+w*3][o*nF*3+m*3+1]=integral(e,q*2+1,w*3,0,1,m*3+1,o*2)+integral(e,q*2,w*3+1,5,5,m*3,o*2+1)
                        Kel[e*bnod*nF*3+q*nF*3+w*3][o*nF*3+m*3+2]=integral(e,q*2+1,w*3,0,2,m*3+2,o*2)+integral(e,q*2,w*3+2,3,3,m*3,o*2+1)
                        Kel[e*bnod*nF*3+q*nF*3+w*3+1][o*nF*3+m*3]=integral(e,q*2,w*3+1,1,0,m*3,o*2+1)+integral(e,q*2+1,w*3,5,5,m*3+1,o*2)
                        Kel[e*bnod*nF*3+q*nF*3+w*3+1][o*nF*3+m*3+1]=integral(e,q*2+1,w*3,5,5,m*3,o*2+1)+integral(e,q*2,w*3+1,1,1,m*3+1,o*2)+integral(e,q*2,w*3+2,4,4,m*3+2,o*2)
                        Kel[e*bnod*nF*3+q*nF*3+w*3+1][o*nF*3+m*3+2]=integral(e,q*2,w*3+1,1,2,m*3+2,o*2)+integral(e,q*2,w*3+2,4,4,m*3+1,o*2)
                        Kel[e*bnod*nF*3+q*nF*3+w*3+2][o*nF*3+m*3]=integral(e,q*2,w*3+2,2,0,m*3,o*2+1)+integral(e,q*2+1,w*3,3,3,m*3+2,o*2)
                        Kel[e*bnod*nF*3+q*nF*3+w*3+2][o*nF*3+m*3+1]=integral(e,q*2,w*3+2,2,1,m*3+1,o*2)+integral(e,q*2,w*3+1,4,4,m*3+2,o*2)
                        Kel[e*bnod*nF*3+q*nF*3+w*3+2][o*nF*3+m*3+2]=integral(e,q*2+1,w*3,3,3,m*3,o*2+1)+integral(e,q*2,w*3+1,4,4,m*3+1,o*2)+integral(e,q*2,w*3+2,2,2,m*3+2,o*2)

                            
    return

"""Finally I will call function integraltot() to build up matrix Kel"""
integraltot()

"""Now, if multi-Lagrange elements have been used to discretise the cross-section, we have to build the stiffeness matrix of a longitudinal element Kel1 from Kel"""
Kel1=np.zeros((bnod*nn_t*3,bnod*nn_t*3))
Mel1=np.zeros((bnod*nn_t*3,bnod*nn_t*3))    

for e in range(0,ne_t):
    for i in range(0,bnod):
        for j in range(0,bnod):
            for q in range(0,nF):
                for r in range(0,nF):
                    for s in range(0,3):              
                        for m in range(0,3):
                            Kel1[i*nn_t*3+(Ctrav[e][q]-1)*3+s][j*nn_t*3+(Ctrav[e][r]-1)*3+m]=Kel1[i*nn_t*3+(Ctrav[e][q]-1)*3+s][j*nn_t*3+(Ctrav[e][r]-1)*3+m]+Kel[e*bnod*nF*3+i*nF*3+q*3+s][j*nF*3+r*3+m]
                            
""" Now I build the mass matrix """

Mel=[]

def Mintegrand(beta,alpha, eta,e,c,b,k,p):
        global F
        global N
        
        
        F.clear()
        F1=f1t(alpha,beta,Nod[0][0],Nod[0][1])
        F2=f2t(alpha,beta,Nod[1][0],Nod[1][1])
        F3=f1t(alpha,beta,Nod[2][0],Nod[2][1])
        F4=f2t(alpha,beta,Nod[3][0],Nod[3][1])
        F5=f1t(alpha,beta,Nod[4][0],Nod[4][1])
        F6=f2t(alpha,beta,Nod[5][0],Nod[5][1])
        F7=f1t(alpha,beta,Nod[6][0],Nod[6][1])
        F8=f2t(alpha,beta,Nod[7][0],Nod[7][1])
        F9=f3t(alpha,beta,Nod[8][0],Nod[8][1])
        

        F.append(F1)
        F.append(F2)
        F.append(F3)
        F.append(F4)
        F.append(F5)
        F.append(F6)
        F.append(F7)
        F.append(F8)
        F.append(F9)
        
        
        N.clear()
    
        #decomment the following two raws if bnod=2
#        N.append((1/2)*(1-(x/(le/2))))
#        N.append((1/2)*(1+(x/(le/2))))    
#      
        #decomment the following 4 raws if bnod=4
#        
#        N.append((-9/16)*(eta+1/3)*(eta-1/3)*(eta-1))
#        N.append((27/16)*(eta+1)*(eta-1/3)*(eta-1))
#        N.append((-27/16)*(eta+1)*(eta+1/3)*(eta-1))
#        N.append((9/16)*(eta+1/3)*(eta-1/3)*(eta+1))          
#        
        
    #decomment the following 3 raws if bnod=3
        
        N.append((1/2)*eta*(eta-1))
        N.append(+(1+eta)*(1-eta))
        N.append((1/2)*eta*(eta+1))
    
        return N[c]*F[b]*F[k]*N[p]*rho*Jac(e,alpha,beta)*le/2
  

def Mintegral(e,c,b,k,p):
    xi=[0,-0.774596669241483,+0.7745966692411483]
    w=[0.888888888888889,0.555555555555556,0.555555555555556]
    
    I=0
    for i in range(0,len(w)):
        for j in range(0,len(w)):
            for m in range(0,len(w)):
                I=I+(Mintegrand(xi[i],xi[j],xi[m],e,c,b,k,p))*(w[i])*(w[j])*(w[m])
                
    return I

def Mintegraltot():
    global Mel
    Mel=np.zeros((ne_t*bnod*nF*3,bnod*nF*3))
    for e in range(0,ne_t):
        for q in range(0,bnod):
            for o in range(0,bnod):
                for w in range(0,nF):
                    for m in range(0,nF):
                        Mel[e*bnod*nF*3+q*nF*3+w*3][o*nF*3+m*3]=Mintegral(e,q,w,m,o)
                        Mel[e*bnod*nF*3+q*nF*3+w*3+1][o*nF*3+m*3+1]=Mintegral(e,q,w,m,o)
                        Mel[e*bnod*nF*3+q*nF*3+w*3+2][o*nF*3+m*3+2]=Mintegral(e,q,w,m,o)

                    
Mintegraltot()

for e in range(0,ne_t):
    for i in range(0,bnod):
        for j in range(0,bnod):
            for q in range(0,nF):
                for r in range(0,nF):
                    for s in range(0,3):              
                        for m in range(0,3):
                            Mel1[i*nn_t*3+(Ctrav[e][q]-1)*3+s][j*nn_t*3+(Ctrav[e][r]-1)*3+m]=Mel1[i*nn_t*3+(Ctrav[e][q]-1)*3+s][j*nn_t*3+(Ctrav[e][r]-1)*3+m]+Mel[e*bnod*nF*3+i*nF*3+q*3+s][j*nF*3+r*3+m]

np.array(Mel)
np.array(Kel)


""" Now I have to build the final stiffness and Mass matrix """

K=np.zeros((nn_x*nn_t*3,nn_x*nn_t*3)) 
M=np.zeros((nn_x*nn_t*3,nn_x*nn_t*3))

""" Now I will assembly the element stiffness matrix Kel """


for el in range(0,ne_x):
    for k in range (0,bnod):
        for s in range(0,bnod):
            for i in range(0,nn_t):
                for j in range(0,nn_t):
                    for m in range(0,3):
                        for r in range(0,3):
                            K[(Cx[el][k])*nn_t*3+i*3+m][(Cx[el][s])*nn_t*3+j*3+r]=K[(Cx[el][k])*nn_t*3+i*3+m][(Cx[el][s])*nn_t*3+j*3+r]+Kel1[k*nn_t*3+i*3+m][s*nn_t*3+j*3+r]
                    

""" Now I will assembly the element mass matrix Mel """       
        
for el in range(0,ne_x):
    for k in range (0,bnod):
        for s in range(0,bnod):
            for i in range(0,nn_t):
                for j in range(0,nn_t):
                    for m in range(0,3):
                        for r in range(0,3):
                            M[(Cx[el][k])*nn_t*3+i*3+m][(Cx[el][s])*nn_t*3+j*3+r]=M[(Cx[el][k])*nn_t*3+i*3+m][(Cx[el][s])*nn_t*3+j*3+r]+Mel1[k*nn_t*3+i*3+m][s*nn_t*3+j*3+r]


eigv=[] 
omega2=[]  
omega=[]
sK=sparse.csr_matrix(K)
sM=sparse.csr_matrix(M)


[omega2,eigv]=eigsh(sK,50,sM,which='SM') #to obrain the first 50 omega_i^2
      
for i in range(0,len(omega2)):
        if -0.001<omega2[i]<0.001:
             omega.append(Ma.sqrt(np.abs(np.real(omega2[i]))))
        else:
            if omega2[i]>=0:
                omega.append(Ma.sqrt(np.real(omega2[i])))



freq=[]

for i in range(0,len(omega)):
    
    freq.append(omega[i]/(2*Ma.pi)) #vector with the fist 50 frequencies
    
    
"""The following functions have been created to represent the eigenvectors"""

def Func(el,w,yq,zq):
    #voglio capire in quale elemento della cross section mi trovo
    lqy=Con[el][4]-Con[el][0]
    lqz=Con[el][8]-Con[el][5]
    alphaq=(yq-Bar[el][0])/(lqy/2)
    betaq=(zq-Bar[el][1])/(lqz/2)
    
    

    
    F.clear()
    F1=f1t(alphaq,betaq,Nod[0][0],Nod[0][1])
    F2=f2t(alphaq,betaq,Nod[1][0],Nod[1][1])
    F3=f1t(alphaq,betaq,Nod[2][0],Nod[2][1])
    F4=f2t(alphaq,betaq,Nod[3][0],Nod[3][1])
    F5=f1t(alphaq,betaq,Nod[4][0],Nod[4][1])
    F6=f2t(alphaq,betaq,Nod[5][0],Nod[5][1])
    F7=f1t(alphaq,betaq,Nod[6][0],Nod[6][1])
    F8=f2t(alphaq,betaq,Nod[7][0],Nod[7][1])
    F9=f3t(alphaq,betaq,Nod[8][0],Nod[8][1])
    
    #it is true just for squared element
    
    F.append(F1)
    F.append(F2)
    F.append(F3)
    F.append(F4)
    F.append(F5)
    F.append(F6)
    F.append(F7)
    F.append(F8)
    F.append(F9)
        
    return F[w]
 
def Nfunc(q,xq):
    N.clear()
        
   
    #if bnod=2--> decomment the following two raws
#    N.append((1/2)*(1-(xq/(le/2))))      
#    N.append((1/2)*(1+(xq/(le/2))))    
    
    #if bnod=4--> decomment the following 5 raws
#    etaq=xq/(le/2)    
#    N.append((-9/16)*(etaq+1/3)*(etaq-1/3)*(etaq-1))
#    N.append((27/16)*(etaq+1)*(etaq-1/3)*(etaq-1))
#    N.append((-27/16)*(etaq+1)*(etaq+1/3)*(etaq-1))
#    N.append((9/16)*(etaq+1/3)*(etaq-1/3)*(etaq+1))

#if bnod=3--> decomment the following 4 raws
    etaq=xq/(le/2)
    N.append((1/2)*etaq*(etaq-1))
    N.append(+(1+etaq)*(1-etaq))
    N.append((1/2)*etaq*(etaq+1))
    
    return N[q]

# this function will provide the index in eigv of the nmin-th frequency
def findindex(nmin):
    
    minfreq=[]
    for i in range(0,len(freq)):
        minfreq.append(freq[i])
    
    for i in range(0,nmin-1):
        del minfreq[minfreq.index(min(minfreq))]
    print(freq[freq.index(min(minfreq))])
    return freq.index(min(minfreq))
    
"""This function will represent the nm-th first eigenvector with a factor scale equal to scale"""    
def plotundeformed():
    
    
    yk=np.array(list(np.arange(-h/2,h/2+0.1,0.1))) 
    zk=np.array(list(np.arange(-h/2,h/2+0.1,0.1))) 
    xk=np.array(list(np.arange(0,L,0.1))) 
    
    X=[]
    Y=[]
    Z=[]
    el=80
    
    
    for i in range(0,len(yk)):
        for j in range(0,len(zk)):
            for k in range(0,len(xk)):
                el=80
                for e in range(0,ne_t):
                    if Con[e][0]<=yk[i]<=Con[e][4] and Con[e][5]<=zk[j]<=Con[e][9]:
                        el=e 
                if el!=80:
                    uy=0
                    uz=0
                    ux=0
                    
                         
                    X.append((xk[k]+ux))
                    Y.append((yk[i]+uy))
                    Z.append((zk[j]+uz))
                    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(Y,X,Z,',',c=X)
    ax.set_title('Deformation');
    ax.set_xlabel('Y Label')
    ax.set_ylabel('X Label')
    ax.set_zlabel('Z Label')
    plt.show()
                     
        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X, Y, Z,',',c=X)
    ax.set_title('Deformation');
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(Y,Z,X,',',c=X)
    ax.set_title('Deformation');
    ax.set_xlabel('Y Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('X Label')
    plt.show()
    
    return
    

def plot3dmode(nm,scale):
    global umg
    
    yk=np.array(list(np.arange(-h/2,h/2+0.1,0.1))) 
    zk=np.array(list(np.arange(-h/2,h/2+0.1,0.1))) 
    xk=np.array(list(np.arange(0,L,0.1))) 
    
    X=[]
    Y=[]
    Z=[]
    el=80
    um=[]
    
    
    for i in range(0,nn_x*nn_t*3):
        um.append(eigv[i][nm])
    
    
        
    
    
        
    for i in range(0,len(yk)):
        for j in range(0,len(zk)):
            for k in range(0,len(xk)):
                el=80
                for e in range(0,ne_t):
                    if Con[e][0]<=yk[i]<=Con[e][4] and Con[e][5]<=zk[j]<=Con[e][9]:
                        el=e 
                if el!=80:
                    uy=0
                    uz=0
                    ux=0
                    
                    q=int(xk[k]/le)
                    for n in range(0,bnod):
                        for f in range(0,nF):
                            
                            uy=uy+Nfunc(n,xk[k]-q*le-le/2)*Func(el,f,yk[i],zk[j])*um[q*(bnod-1)*(nn_t)*3+n*(nn_t)*3+(Ctrav[el][f]-1)*3+1]
                            uz=uz+Nfunc(n,xk[k]-q*le-le/2)*Func(el,f,yk[i],zk[j])*um[q*(bnod-1)*(nn_t)*3+n*(nn_t)*3+(Ctrav[el][f]-1)*3+2]
                            ux=ux+Nfunc(n,xk[k]-q*le-le/2)*Func(el,f,yk[i],zk[j])*um[q*(bnod-1)*(nn_t)*3+n*(nn_t)*3+(Ctrav[el][f]-1)*3+0]
                         
                    
                         
                    X.append((xk[k]+ux))
                    Y.append((yk[i]+uy))
                    Z.append((zk[j]+uz))
                    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(Y,X,Z,',',c=X)
    ax.set_title('Deformation');
    ax.set_xlabel('Y Label')
    ax.set_ylabel('X Label')
    ax.set_zlabel('Z Label')
    plt.show()
                     
        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X, Y, Z,',',c=X)
    ax.set_title('Deformation');
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(Y,Z,X,',',c=X)
    ax.set_title('Deformation');
    ax.set_xlabel('Y Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('X Label')
    plt.show()
    
    return

plot3dmode(findindex(7),50)
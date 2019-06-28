# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:05:53 2019

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
nF=10     #nF is what in the previous sections has been referres as M, that is to say the number of terms for each cross section expansion F_\tau
ne_x=20   #number of longitudinal elements
nn_x=(bnod-1)*ne_x+1  #number of longitudinal nodes
R=1   #External rayon of the cross-section
t=0.1  #thickness of the cross-section walls
Rb=0.6 #external rayon of a booster
tb=0.2  #thickness of a booster
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

"""Let's build the nucleus stiffeness matrix"""

#This funtion will provide the function to be integrated over the beam element volume
def integrand(phi,r,x,dec,c,b,n,a,k,p):
    global F
    F.clear()
    
    y=r*Ma.cos(phi)+dec
    z=r*Ma.sin(phi)
    
    #Functions Fi
    F1=1
    F2=y
    F3=z
    F4=y**2
    F5=y*z
    F6=z**2
    F7=y**3
    F8=(y**2)*z
    F9=y*(z**2)
    F10=z**3
    
   
    #Partial derivatives of Fi
    F1y=0
    F1z=0
    F2y=1
    F2z=0
    F3y=0
    F3z=1
    F4y=2*y
    F4z=0
    F5y=z
    F5z=y
    F6y=0
    F6z=2*z
    F7y=3*(y**2)
    F7z=0
    F8y=2*y*z
    F8z=y**2
    F9y=z**2
    F9z=2*y*z
    F10y=0
    F10z=3*(z**2)
    
    
        
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
    F.append(F10)
    F.append(F10y)
    F.append(F10z)
    
        
    N.clear()

#    if bnod=2--> decomment the following 4 raws
#    N.append((1/2)*(1-(x/(le/2))))
#    N.append(-1/(le))    
#    N.append((1/2)*(1+(x/(le/2))))    
#    N.append(1/(le))  

#    if bnod=3--> decomment the following 7 raws    
    eta=x/(le/2)
    N.append((1/2)*eta*(eta-1))
    N.append((1/le)*(2*eta-1))
    
    N.append(+(1+eta)*(1-eta))
    N.append(-(4/le)*eta)
    
    
    N.append((1/2)*eta*(eta+1))
    N.append((1/le)*(2*eta+1))

#    if bnod=4--> decomment the following 9 raws   
#    eta=x/(le/2)
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
        
    return N[c]*F[b]*C[n][a]*F[k]*N[p]*r

#this function will perform the integral of the function integrand over the beam element
def integral(c,b,n,a,k,p):
    return (integ.tplquad(integrand,-le/2,le/2,lambda x: 0, lambda x: R,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(0,c,b,n,a,k,p),epsabs=100, epsrel=100)[0])-(integ.tplquad(integrand,-le/2,le/2,lambda x: 0, lambda x: R-t,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(0,c,b,n,a,k,p),epsabs=100, epsrel=100)[0])+(integ.tplquad(integrand,-le/2,le/2,lambda x: 0, lambda x: Rb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(Rb+R,c,b,n,a,k,p),epsabs=100, epsrel=100)[0])-(integ.tplquad(integrand,-le/2,le/2,lambda x: 0, lambda x: Rb-tb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(R+Rb,c,b,n,a,k,p),epsabs=100, epsrel=100)[0])+(integ.tplquad(integrand,-le/2,le/2,lambda x: 0, lambda x: Rb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(-(Rb+R),c,b,n,a,k,p),epsabs=100, epsrel=100)[0])-(integ.tplquad(integrand,-le/2,le/2,lambda x: 0, lambda x: Rb-tb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(-(R+Rb),c,b,n,a,k,p),epsabs=100, epsrel=100)[0])

def integraltot():
    global Kel
    Kel=np.zeros((bnod*nF*3,bnod*nF*3))
    
    for q in range(0,bnod):   #loop over the nodes of a longitudinal element
        for o in range(0,bnod):   #loop over the nodes of a longitudinal element
            for w in range(0,nF):   #loop over the terms of transverse expansion over the cross-section
                for m in range(0,nF):   #loop over the terms of transverse expansion over the cross-section
                    Kel[q*nF*3+w*3][o*nF*3+m*3]=integral(q*2+1,w*3,0,0,m*3,o*2+1)+integral(q*2,w*3+1,5,5,m*3+1,o*2)+integral(q*2,w*3+2,3,3,m*3+2,o*2)
                    Kel[q*nF*3+w*3][o*nF*3+m*3+1]=integral(q*2+1,w*3,0,1,m*3+1,o*2)+integral(q*2,w*3+1,5,5,m*3,o*2+1)
                    Kel[q*nF*3+w*3][o*nF*3+m*3+2]=integral(q*2+1,w*3,0,2,m*3+2,o*2)+integral(q*2,w*3+2,3,3,m*3,o*2+1)
                    Kel[q*nF*3+w*3+1][o*nF*3+m*3]=integral(q*2,w*3+1,1,0,m*3,o*2+1)+integral(q*2+1,w*3,5,5,m*3+1,o*2)
                    Kel[q*nF*3+w*3+1][o*nF*3+m*3+1]=integral(q*2+1,w*3,5,5,m*3,o*2+1)+integral(q*2,w*3+1,1,1,m*3+1,o*2)+integral(q*2,w*3+2,4,4,m*3+2,o*2)
                    Kel[q*nF*3+w*3+1][o*nF*3+m*3+2]=integral(q*2,w*3+1,1,2,m*3+2,o*2)+integral(q*2,w*3+2,4,4,m*3+1,o*2)
                    Kel[q*nF*3+w*3+2][o*nF*3+m*3]=integral(q*2,w*3+2,2,0,m*3,o*2+1)+integral(q*2+1,w*3,3,3,m*3+2,o*2)
                    Kel[q*nF*3+w*3+2][o*nF*3+m*3+1]=integral(q*2,w*3+2,2,1,m*3+1,o*2)+integral(q*2,w*3+1,4,4,m*3+2,o*2)
                    Kel[q*nF*3+w*3+2][o*nF*3+m*3+2]=integral(q*2+1,w*3,3,3,m*3,o*2+1)+integral(q*2,w*3+1,4,4,m*3+1,o*2)+integral(q*2,w*3+2,2,2,m*3+2,o*2)
                            
                            
    return

"""Finally I will call function integraltot() to build up matrix Kel"""
integraltot()

""" Now I build the mass matrix """

Mel=[]

def Mintegrand(phi,r,x,dec,c,b,k,p):
        global F
        global N
        
        y=r*Ma.cos(phi)+dec
        z=r*Ma.sin(phi) 
        
        F.clear()
        F1=1
        F2=y
        F3=z
        F4=y**2
        F5=y*z
        F6=z**2
        F6=z**2
        F7=y**3
        F8=(y**2)*z
        F9=y*(z**2)
        F10=z**3
        

        
        F.append(F1)
        F.append(F2)
        F.append(F3)
        F.append(F4)
        F.append(F5)
        F.append(F6)
        F.append(F7)
        F.append(F8)
        F.append(F9)
        F.append(F10)
        
        
        
       
        N.clear()
    
        #decomment the following two raws if bnod=2
#        N.append((1/2)*(1-(x/(le/2))))
#        N.append((1/2)*(1+(x/(le/2))))    
#      
        #decomment the following 5 raws if bnod=4
#        eta=x/(le/2)
#        N.append((-9/16)*(eta+1/3)*(eta-1/3)*(eta-1))
#        N.append((27/16)*(eta+1)*(eta-1/3)*(eta-1))
#        N.append((-27/16)*(eta+1)*(eta+1/3)*(eta-1))
#        N.append((9/16)*(eta+1/3)*(eta-1/3)*(eta+1))          
#        
        
    #decomment the following 4 raws if bnod=3
        eta=x/(le/2)
        N.append((1/2)*eta*(eta-1))
        N.append(+(1+eta)*(1-eta))
        N.append((1/2)*eta*(eta+1))
    
        return N[c]*F[b]*F[k]*N[p]*rho*r
  

def Mintegral(c,b,k,p):
    return (integ.tplquad(Mintegrand,-le/2,le/2,lambda x: 0, lambda x: R,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(0,c,b,k,p),epsabs=1.49, epsrel=1.49)[0])-(integ.tplquad(Mintegrand,-le/2,le/2,lambda x: 0, lambda x: R-t,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(0,c,b,k,p),epsabs=1.49, epsrel=1.49)[0])+(integ.tplquad(Mintegrand,-le/2,le/2,lambda x: 0, lambda x: Rb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(Rb+R,c,b,k,p),epsabs=1.49, epsrel=1.49)[0])-(integ.tplquad(Mintegrand,-le/2,le/2,lambda x: 0, lambda x: Rb-tb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(R+Rb,c,b,k,p),epsabs=1.49, epsrel=1.49)[0])+(integ.tplquad(Mintegrand,-le/2,le/2,lambda x: 0, lambda x: Rb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(-(Rb+R),c,b,k,p),epsabs=1.49, epsrel=1.49)[0])-(integ.tplquad(Mintegrand,-le/2,le/2,lambda x: 0, lambda x: Rb-tb,lambda x,r: 0, lambda x,r:2*Ma.pi, args=(-(R+Rb),c,b,k,p),epsabs=1.49, epsrel=1.49)[0])

def Mintegraltot():
    global Mel
    Mel=np.zeros((bnod*nF*3,bnod*nF*3))
    for q in range(0,bnod):
        for o in range(0,bnod):
            for w in range(0,nF):
                for m in range(0,nF):
                    Mel[q*nF*3+w*3][o*nF*3+m*3]=Mintegral(q,w,m,o)
                    Mel[q*nF*3+w*3+1][o*nF*3+m*3+1]=Mintegral(q,w,m,o)
                    Mel[q*nF*3+w*3+2][o*nF*3+m*3+2]=Mintegral(q,w,m,o)
                    
                    
Mintegraltot()

np.array(Mel)
np.array(Kel)


""" Now I have to build the final stiffness and Mass matrix """

K=np.zeros((nn_x*nF*3,nn_x*nF*3)) 
M=np.zeros((nn_x*nF*3,nn_x*nF*3))


""" Now I will assembly the element stiffness matrix Kel """


for el in range(0,ne_x):
    for k in range (0,bnod):
        for s in range(0,bnod):
            for i in range(0,nF):
                for j in range(0,nF):
                    for m in range(0,3):
                        for r in range(0,3):
                            K[(Cx[el][k])*nF*3+i*3+m][(Cx[el][s])*nF*3+j*3+r]=K[(Cx[el][k])*nF*3+i*3+m][(Cx[el][s])*nF*3+j*3+r]+Kel[k*nF*3+i*3+m][s*nF*3+j*3+r]
      


""" Now I will assembly the element mass matrix Mel """       
for el in range(0,ne_x):
    for k in range (0,bnod):
        for s in range(0,bnod):
            for i in range(0,nF):
                for j in range(0,nF):
                    for m in range(0,3):
                        for r in range(0,3):
                            M[(Cx[el][k])*nF*3+i*3+m][(Cx[el][s])*nF*3+j*3+r]=M[(Cx[el][k])*nF*3+i*3+m][(Cx[el][s])*nF*3+j*3+r]+Mel[k*nF*3+i*3+m][s*nF*3+j*3+r]

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

def Func(w,yq,zq):
    F.clear()
    
    
    F1=1
    F2=yq
    F3=zq
    F4=yq**2
    F5=yq*zq
    F6=zq**2
    F7=yq**3
    F8=(yq**2)*zq
    F9=yq*(zq**2)
    F10=zq**3
    

    
    F.append(F1)
    F.append(F2)
    F.append(F3)
    F.append(F4)
    F.append(F5)
    F.append(F6)
    F.append(F7)
    F.append(F8)
    F.append(F9)
    F.append(F10)
    
    
        
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
def plot3dmode(nm,scale):
  
    
    yk=np.array(list(np.arange(-(R+2*Rb),R+2*Rb+0.05,0.05))) 
    zk=np.array(list(np.arange(-R,R+0.05,0.05))) 
    xk=np.array(list(np.arange(0,L+0.1,0.1))) 
    
    
    X=[]
    Y=[]
    Z=[]
    
    um=[]
    
   
    for i in range(0,nn_x*nF*3):
        um.append(eigv[i][nm])
    
    
        
    for i in range(0,len(yk)):
        for j in range(0,len(zk)):
            for k in range(0,len(xk)-1):
                q=int(xk[k]/le)
                if -R<=yk[i]<=R:
                    de=0
                    Rmin=R-t
                    Rmax=R
                elif yk[i]>R:
                    de=R+Rb
                    Rmin=Rb-tb
                    Rmax=Rb
                else:
                    de=-(R+Rb)
                    Rmin=Rb-tb
                    Rmax=Rb
                    
                rk=Ma.sqrt((yk[i]-de)**2+zk[j]**2)
                
                if Rmin<=rk<=Rmax:
                    uy=0
                    uz=0
                    ux=0
            
                    for n in range(0,bnod):
                        for f in range(0,nF):
                            uy=uy+Nfunc(n,xk[k]-q*le-le/2)*Func(f,yk[i],zk[j])*um[q*(bnod-1)*(nF)*3+n*(nF)*3+f*3+1]
                            uz=uz+Nfunc(n,xk[k]-q*le-le/2)*Func(f,yk[i],zk[j])*um[q*(bnod-1)*(nF)*3+n*(nF)*3+f*3+2]
                            ux=ux+Nfunc(n,xk[k]-q*le-le/2)*Func(f,yk[i],zk[j])*um[q*(bnod-1)*(nF)*3+n*(nF)*3+f*3+0]
                    
                
                    X.append((xk[k]+np.real(ux)*scale))
                    Y.append((yk[i]+np.real(uy)*scale))
                    Z.append((zk[j]+np.real(uz)*scale))
                
                    
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



def plotundeformed():
    
    
    yk=np.array(list(np.arange(-(R+2*Rb),R+2*Rb+0.05,0.05))) 
    zk=np.array(list(np.arange(-R,R+0.05,0.05))) 
    xk=np.array(list(np.arange(0,L+0.1,0.1))) 
    
    
    X=[]
    Y=[]
    Z=[]
   
        
    for i in range(0,len(yk)):
        for j in range(0,len(zk)):
            for k in range(0,len(xk)-1):
                
                if -R<=yk[i]<=R:
                    de=0
                    Rmin=R-t
                    Rmax=R
                elif yk[i]>R:
                    de=R+Rb
                    Rmin=Rb-tb
                    Rmax=Rb
                else:
                    de=-(R+Rb)
                    Rmin=Rb-tb
                    Rmax=Rb
                    
                rk=Ma.sqrt((yk[i]-de)**2+zk[j]**2)
                
                if Rmin<=rk<=Rmax:
                    
                    X.append((xk[k]))
                    Y.append((yk[i]))
                    Z.append((zk[j]))
                    
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
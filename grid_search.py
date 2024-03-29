import numpy as np
import matplotlib.pyplot as plt
#assume a polynomial of the form
#c4vd^4+c3vd^3+c2vd^2+c1vd+c0
#define the known values
from numpy import random
import time
B0=6
L0=6
phi=np.pi/3
vb=0.20
dphi_dt=-0.02
vd=2
phi_act = 60*(np.pi/180)
hist_array=np.zeros((181,360))
roots_array = np.zeros(181*4)
t=0.1
tstart=time.time()
for angle in range(0,360,1):
    phi=(angle/180)*np.pi
    
    for i in range(1,182):
        t=3*i*(86*1e-6)

        g=B0*np.cos(phi)+L0*np.sin(phi)-2*vb*np.sin(phi)*t
        h=L0-vb*t
        k=2*B0*np.cos(phi)*t+2*np.sin(phi)*t*h
        dr_dt=( (B0+vd*np.cos(phi_act)*t)*vd*np.cos(phi_act) + (L0+vd*np.sin(phi_act)*t-vb*t)*(vd*np.sin(phi_act)-vb) )/(np.sqrt( np.square(B0+vd*np.cos(phi_act)*t) +np.square(L0+vd*np.sin(phi_act)*t-vb*t)        ))+ random.normal(0,0.037)
        sq=dr_dt*dr_dt
    
        c4=t*t
        c3=2*t*g
        c2=g*g-2*t*h*vb-sq*t*t
        c1=-2*g*h*vb-sq*k
        c0=h*h*vb*vb-sq*(B0*B0+h*h)
        coeff=[c4,c3, c2, c1, c0]
        
        arr = np.roots(coeff)
        roots_array[i-1:i+3]=np.abs(arr)
        if(np.imag(arr[2])!=0):
            if angle==0:
                continue
            hist_array[i-1][angle+0]=hist_array[i-1][angle-1]
            continue
        # if angle==60:
            # print(arr)
        hist_array[i-1][angle+0]=np.real(arr[2])
    # if angle==60:
        # plt.scatter(roots_array, np.zeros(len(roots_array)))
        # plt.show()

print(time.time()-tstart)
variance=np.zeros(360)
for i in range(0,360):
    dummy=hist_array[:,i]
    variance[i]=np.var(dummy)

print(np.argmin(variance))
plt.plot(variance)
plt.show()




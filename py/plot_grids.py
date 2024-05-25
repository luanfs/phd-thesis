import numpy as np
import matplotlib.pyplot as plt


def area(N, gtype):
    # some constants
    figformat='png'

    r = 1
    a = r/np.sqrt(2.0)
    a2 = a*a
    erad =  6371.20

    if gtype==0:
       aref = np.arcsin(1.0/np.sqrt(3.0))
       rref = np.sqrt(2.0)

       x = np.linspace(-aref, aref, N)
       y = np.linspace(-aref, aref, N)
       dx = x[1]-x[0]
       dy = y[1]-y[0]
       x, y = np.meshgrid(x,y)

       fx = rref*np.tan(x)
       fy = rref*np.tan(y)

       dfx = rref/np.cos(x)**2
       dfy = rref/np.cos(y)**2

    elif gtype==1:
       aref = 1.0
       rref = 1.0

       x = np.linspace(-aref, aref, N)
       y = np.linspace(-aref, aref, N)
       dx = x[1]-x[0]
       dy = y[1]-y[0]
       x, y = np.meshgrid(x,y)
     
       fx = x
       fy = y

       dfx = np.ones(np.shape(fx))
       dfy = np.ones(np.shape(fy))

    elif gtype == 2:
       aref = np.pi*0.25
       rref = 1.0

       x = np.linspace(-aref, aref, N)
       y = np.linspace(-aref, aref, N)
       dx = x[1]-x[0]
       dy = y[1]-y[0]
       x, y = np.meshgrid(x,y)

       fx = rref*np.tan(x)
       fy = rref*np.tan(y)

       dfx = rref/np.cos(x)**2
       dfy = rref/np.cos(y)**2

    mt = dfx*dfy/(1+fx*fx+fy*fy)**1.5
    area = erad*erad*mt*dx*dy
    r = 2*np.sqrt(area/np.pi)
    print(N,np.mean(area),np.amin(area),np.amax(area),np.amax(area)/np.amin(area))
    #print(N,np.mean(r),np.amin(r),np.amax(r),np.amax(r)/np.amin(r))
Ns = (48,96,192,384,768)
gtype = (0,1,2)

for g in gtype:
    for N in Ns:
        area(N, g)
    print()

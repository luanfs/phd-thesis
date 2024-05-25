import numpy as np
import matplotlib.pyplot as plt


# Python script to plot the outputs
# directory
nbfaces = 4
figformat='png'

# some constants
N = 10 # number of cells

gtype = 0
r = 1
a = r/np.sqrt(2.0)
a2 = a*a

if gtype==0:
   aref = np.arcsin(1.0/np.sqrt(3.0))
   rref = np.sqrt(2.0)
   x = np.linspace(-aref, aref, N)
   fx = rref*np.tan(x)

elif gtype==1:
   aref = 1.0
   rref = 1.0
   x = np.linspace(-aref, aref, N)
   fx = x

elif gtype == 2:
   aref = np.pi*0.25
   rref = 1.0
   x = np.linspace(-aref, aref, N)
   fx = rref*np.tan(x)

fx2 = fx*fx
d = np.sqrt(1 + fx2)

dpi=100
plt.figure(figsize=(1000/dpi, 1000/dpi), dpi=dpi)
plt.plot([-a,a] ,[a,a]  ,color='blue')
plt.plot([-a,-a] ,[a,-a]  ,color='green')
plt.plot([a,a] ,[a,-a]  ,color='red')
plt.plot([a,-a] ,[-a,-a]  ,color='purple')

X = 1/d
Y = fx/d

#plt.axhline(0, color='black')
#plt.axvline(0, color='black')

for i in range(0,N):
   plt.plot(-X[i], Y[i],'o',color='green')
   plt.plot( Y[i],-X[i],'o',color='purple')
   plt.plot( X[i], Y[i],'o',color='red')
   plt.plot( Y[i], X[i],'o',color='blue')
   plt.plot( a*fx[i], a,'s',color='blue')
   plt.plot([Y[i],0],[X[i], 0],'--',color='blue')

for i in range(0,N-1):
   plt.plot( [ Y[i], Y[i+1]], [ X[i], X[i+1]],color='blue')
   plt.plot( [-X[i],-X[i+1]], [ Y[i], Y[i+1]],color='green')
   plt.plot( [ X[i], X[i+1]], [ Y[i], Y[i+1]],color='red')
   plt.plot( [ Y[i], Y[i+1]], [-X[i],-X[i+1]],color='purple')
   print(i,Y[i],X[i])


plt.xlabel('$p_x$',fontsize=25)
plt.ylabel('$p_y$',fontsize=25) 
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.savefig('g'+str(gtype)+'.'+figformat, format=figformat)

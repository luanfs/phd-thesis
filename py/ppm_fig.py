import numpy as np
import matplotlib.pyplot as plt


# Python script to plot the outputs
# directory
figformat='png'
pi = np.pi

# some constants
N = 8 # number of cells
L = 6371
Lo2 = L*0.5

def qfield(x):
   q = 0.9*np.exp(-10*(np.sin(-pi*x/L))**2) +0.1 #\
   #+ 0.5*np.exp(-10*(np.sin(-2*pi*x/L))**2)\
   #+ 0.25*np.exp(-10*(np.sin(-4*pi*x/L))**2)\
   #+ 0.5*np.exp(-10*(np.sin(-5*pi*x/L))**2) + 0.1#\
   #+ 1.5*np.exp(-10*(np.sin(-10*pi*x/L))**2) + 0.1
   return q



####################################################################################
# Given the average values of a scalar field Q, this routine constructs
# a piecewise parabolic aproximation of Q using its average value.
####################################################################################
def ppm_reconstruction(Q, N, recon_name):
    i0 = 3
    iend = N+3
    Q_edges = np.zeros(N+7)
    q_L = np.zeros(N+6)
    q_R = np.zeros(N+6)
    dq = np.zeros(N+6)
    q6 = np.zeros(N+6)
    bL = np.zeros(N+6)
    bR = np.zeros(N+6)
    dQ      = np.zeros(N+6)
    dQ_min  = np.zeros(N+6)
    dQ_max  = np.zeros(N+6)
    dQ_mono = np.zeros(N+6)
    if recon_name == 'unlim': # PPM from CW84 paper
        # Values of Q at right edges (q_(j+1/2)) - Formula 1.9 from Collela and Woodward 1984
        Q_edges[i0-1:iend+2] = (7.0/12.0)*(Q[i0-1:iend+2] + Q[i0-2:iend+1]) - (Q[i0:iend+3] + Q[i0-3:iend])/12.0
        # Assign values of Q_R and Q_L
        q_R[i0-1:iend+1] = Q_edges[i0:iend+2]
        q_L[i0-1:iend+1] = Q_edges[i0-1:iend+1]

    elif recon_name == 'mono':  #PPM with monotonization from Lin 04 paper
        # Formula B1 from Lin 04
        dQ[i0-2:iend+2] = 0.25*(Q[i0-1:iend+3] - Q[i0-3:iend+1])
        dQ_min[i0-2:iend+2]  = np.maximum(np.maximum(Q[i0-3:iend+1], Q[i0-2:iend+2]), Q[i0-1:iend+3]) - Q[i0-2:iend+2]
        dQ_max[i0-2:iend+2]  = Q[i0-2:iend+2] - np.minimum(np.minimum(Q[i0-3:iend+1], Q[i0-2:iend+2]), Q[i0-1:iend+3])
        dQ_mono[i0-2:iend+2] = np.minimum(np.minimum(abs(dQ[i0-2:iend+2]), dQ_min[i0-2:iend+2]), dQ_max[i0-2:iend+2]) * np.sign(dQ[i0-2:iend+2])
        #dQ_mono[i0-2:iend+2] = dQ[i0-2:iend+2]

        # Formula B2 from Lin 04
        Q_edges[i0-1:iend+2] = 0.5*(Q[i0-1:iend+2] + Q[i0-2:iend+1]) - (dQ_mono[i0-1:iend+2] - dQ_mono[i0-2:iend+1])/3.0

        # Assign values of Q_R and Q_L
        q_R[i0-1:iend+1] = Q_edges[i0:iend+2]
        q_L[i0-1:iend+1] = Q_edges[i0-1:iend+1]

        # Formula B3 from Lin 04
        q_L[i0-1:iend+1] = Q[i0-1:iend+1] - np.minimum(2.0*abs(dQ_mono[i0-1:iend+1]), abs(q_L[i0-1:iend+1]-Q[i0-1:iend+1])) * np.sign(2.0*dQ_mono[i0-1:iend+1])

        # Formula B4 from Lin 04
        q_R[i0-1:iend+1] = Q[i0-1:iend+1] + np.minimum(2.0*abs(dQ_mono[i0-1:iend+1]), abs(q_R[i0-1:iend+1]-Q[i0-1:iend+1])) * np.sign(2.0*dQ_mono[i0-1:iend+1])

    # Compute the polynomial coefs
    # q(x) = q_L + z*(dq + q6*(1-z)) z in [0,1]
    dq[i0-1:iend+1] = q_R[i0-1:iend+1] - q_L[i0-1:iend+1]
    q6[i0-1:iend+1] = 6*Q[i0-1:iend+1] - 3*(q_R[i0-1:iend+1] + q_L[i0-1:iend+1])
    bL[i0-1:iend+1] = q_L[i0-1:iend+1]-Q[i0-1:iend+1]
    bR[i0-1:iend+1] = q_R[i0-1:iend+1]-Q[i0-1:iend+1]
    print(np.amax(abs(-q6-3.0*(bL+bR))),recon_name)
    return q_L, q_R, dq, q6, bL, bR

######################################################################################################################
colors = ('blue','lime','purple','cyan','magenta','gold','green','orange','red','gray',)
X = np.linspace(-Lo2, Lo2, N+1)
Xc = (X[0:N]+X[1:])*0.5

qc = np.zeros(N+6)
qe = np.zeros(N+7)

qc[3:N+3] = qfield(Xc)
qe[3:N+4] = qfield(X)

#periodic bc
qc[0:3]  = qc[N:N+3]
qc[N+3:] = qc[3:6]
qe[0:3]  = qe[N+1:N+4]
qe[N+4:] = qe[3:6]

qL_u, qR_u, dq_u, q6_u, bL_u, bR_u = ppm_reconstruction(qc, N, recon_name='unlim')
qL_m, qR_m, dq_m, q6_m, bL_m, bR_m = ppm_reconstruction(qc, N, recon_name='mono')
#error = abs(qe[3:N+4]-(qc[2:N+3]+qc[3:N+4])*0.5)
#exit()

error = max(np.amax(abs(qR_u[3:N+3]-qe[4:N+4])), np.amax(abs(qL_u[3:N+3]-qe[3:N+3])))
print(np.amax(abs(error)))
#exit()
emax=0
for k in range(0,N):
   xL, xR = X[k], X[k+1]
   xc = (xL+xR)*0.5
   x = np.linspace(xL,xR,100)
   qexact = qfield(x)
   q_constant =  qc[k+3]*np.ones(np.shape(x))

   #qc = qfield(xc)
   a1u = qL_u[k+3]
   a2u = -(4.0*bL_u[k+3]+2.0*bR_u[k+3])
   a3u = 3.0*(bL_u[k+3]+bR_u[k+3])
   a1m = qL_m[k+3]
   a2m = -(4.0*bL_m[k+3]+2.0*bR_m[k+3])
   a3m = 3.0*(bL_m[k+3]+bR_m[k+3])
   z = (x-xL)/(xR-xL)
   #print(z)
   #qppm_m = qL_m[k+3]  + dq_m[k+3]*z + q6_m[k+3]*z*(1.0-z)
   #qppm_u = qL_u[k+3]  + dq_u[k+3]*z + q6_u[k+3]*z*(1.0-z)

   qppm_m = a1m + a2m*z + a3m*z*z
   qppm_u = a1u + a2u*z + a3u*z*z
   #qppm_u = a1u + a2u*z + a3u*z*z
   #print(k,abs(-q6_m[k+3]-a3m),abs(-q6_u[k+3]-a3u))
  # print(k,abs((q6_m[k+3]+dq_m[k+3]) - a2m))
   #print(-q6_u[k+3]-3.0*(bL_u[k+3]+bR_u[k+3]))
   #print()
   # ppm approximation
   emax = max(emax,np.amax(abs(qppm_u-qexact)))
   #color = colors[k]
   #color = 'blue'

   if k<N-1:
     #plt.plot(x, q_constant, color = 'blue')
     plt.plot(x, qppm_u, color = 'orange', zorder=11)
     #plt.plot(x, qppm_m, color = 'red', zorder=11)
     plt.plot(x, qexact, color = 'blue', linestyle = 'dashed', zorder=10, linewidth=0.8)
   else:
     #plt.plot(x, q_constant, color = 'blue', label='Average values')
     plt.plot(x, qppm_u, color = 'orange', zorder=11, label='Unlimited PPM')
     #plt.plot(x, qppm_m, color = 'red', zorder=11, label='Monotonic PPM')
     plt.plot(x, qexact, color = 'blue', linestyle = 'dashed', zorder=10, linewidth=0.8, label='Exact scalar field')
   
   plt.ylim(0.0,1.2)

 
   # Add vertical lines at subdomain boundaries
   plt.axvline(xL, color='black', linestyle='--', linewidth=0.5)
   plt.axvline(xR, color='black', linestyle='--', linewidth=0.5)

# Label
plt.xlabel('$x$ (km)')
plt.ylabel('$y$')

#plt.plot(Xc, qc[3:N+3] , color = 'blue', marker = 'o', linestyle='None',zorder=11)
plt.plot(Xc, qc[3:N+3] , color = 'blue', marker = 'o', linestyle='None',zorder=11, label='Average values')
plt.legend()
plt.savefig('ppm_reconu', format=figformat)
#plt.show()
print(emax)
#print()
#print(emax)

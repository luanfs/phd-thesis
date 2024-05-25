import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# some constants
nbfaces = 6 # number of faces in a cube
rad2deg = 180.0/np.pi
deg2rad = 1.0/rad2deg
graphdir='../graphs/'

####################################################################################
# Convert from cartesian (x,y,z) to spherical coordinates (lat,lon)
# on the unit sphere.
# Outputs: latitude (lat), longitude (lon)
####################################################################################
def cart2sph(X, Y, Z):
    hypotxy = np.hypot(X, Y)
    lat = np.arctan2(Z, hypotxy)
    lon = np.arctan2(Y, X)
    return lon, lat


####################################################################################
#  General spherical points structure
####################################################################################
class point:
    def __init__(self, N, M, ng):
        # Cartesian coordinates - represented in capital letters (X,Y,Z)
        self.X   = np.zeros((N+2*ng, M+2*ng, nbfaces))
        self.Y   = np.zeros((N+2*ng, M+2*ng, nbfaces))
        self.Z   = np.zeros((N+2*ng, M+2*ng, nbfaces))

        # Spherical coordinates
        self.lat = np.zeros((N+2*ng, M+2*ng, nbfaces))
        self.lon = np.zeros((N+2*ng, M+2*ng, nbfaces))

        # Local coordinates
        self.x = np.zeros(N+2*ng)
        self.y = np.zeros(M+2*ng)

# The panels are indexed as below following Ronchi et al (96).
#      +---+
#      | 4 |
#  +---+---+---+---+
#  | 3 | 0 | 1 | 2 |
#  +---+---+---+---+
#      | 5 |
#      +---+

####################################################################################
#
# This routine computes the Gnomonic mapping based on the equiangular projection
# defined by Ronchi et al (96) for each panel.
# - x, y are 1d arrays storing the angular variables defined in [-aref,aref].
# - The projection is applied on the points (x, y)
# - N is the number of cells along a coordinate axis
# - Returns the Cartesian (X,Y,Z) and spherical (lon, lat) coordinates of the
# projected points.
#
####################################################################################
def equiangular_gnomonic_map(x1, y1, N, M, ng, R_ref):
    # Cartesian coordinates of the projected points
    X = np.zeros((N+2*ng, M+2*ng, nbfaces))
    Y = np.zeros((N+2*ng, M+2*ng, nbfaces))
    Z = np.zeros((N+2*ng, M+2*ng, nbfaces))

    # Creates a grid in [-aref,aref]x[-aref,aref] using
    # the given values of x and y
    x, y = np.meshgrid(x1, y1, indexing='ij')

    # Auxiliary variables
    tanx = R_ref*np.tan(x)
    tany = R_ref*np.tan(y)
    D2   = 1.0 + tanx**2 + tany**2
    D    = np.sqrt(D2)
    invD = 1.0/D
    XoD  = invD*tanx
    YoD  = invD*tany

    # Compute the Cartesian coordinates for each panel
    # with the aid of the auxiliary variables

    # Panel 0
    X[:,:,0] = invD
    Y[:,:,0] = XoD  # x*X
    Z[:,:,0] = YoD  # x*Y

    # Panel 1
    Y[:,:,1] =  invD
    X[:,:,1] = -XoD #-y*X
    Z[:,:,1] =  YoD # y*Y

    # Panel 2
    X[:,:,2] = -invD
    Y[:,:,2] = -XoD # x*X
    Z[:,:,2] =  YoD #-x*Y

   # Panel 3
    Y[:,:,3] = -invD
    X[:,:,3] =  XoD #-y*X
    Z[:,:,3] =  YoD #-y*Y

    # Panel 4
    Z[:,:,4] =  invD
    X[:,:,4] = -YoD #-z*Y
    Y[:,:,4] =  XoD # z*X

    # Panel 5
    Z[:,:,5] = -invD
    X[:,:,5] =  YoD #-z*Y
    Y[:,:,5] =  XoD #-z*X

    # Convert to spherical coordinates
    lon, lat = cart2sph(X, Y, Z)

    return X, Y, Z, lon, lat


####################################################################################
# Compute the metric tensor of the cubed-sphere mapping
# Inputs: 1d arrays x and y containing the cube face coordinates, the cube
# projection and sphere radius R.
# Output: metric tensor G on the grid (x,y)
# Reference: Rancic et al 1996
####################################################################################
def metric_tensor(x1, y1, R):
    x, y = np.meshgrid(x1, y1, indexing='ij')
    tanx, tany = np.tan(x), np.tan(y)
    r = np.sqrt(1 + tanx*tanx + tany*tany)
    G = R*R/r**3
    G = G/(np.cos(x)*np.cos(y))**2
    return G

####################################################################################
#  General cubed sphere structure
####################################################################################
class CS:
    def __init__(self, N, gtype):
        # some constants
        ng = 3
        self.ng = ng # halo size 
        self.N = N
        self.Nt = self.N + 2*self.ng
        Nt = self.Nt

        # indexes
        self.IS, self.ie = self.ng, self.N+self.ng
        self.js, self.je = self.IS, self.ie
        self.isd, self.ied = self.IS-self.ng, self.ie+self.ng
        self.jsd, self.jed = self.isd, self.ied

        # grid type
        if gtype == 'equiangular':
            a_ref = np.pi*0.25
            R_ref = 1.0
        elif gtype == 'equiedge':
            a_ref = np.arcsin(1.0/np.sqrt(3.0))
            R_ref = np.sqrt(2.0)

        self.a_ref = a_ref
        self.R_ref = R_ref
        self.name = gtype

        # Local grid spacing
        dx = 2.0*a_ref/N
        dy = dx
        dxo2, dyo2 = dx*0.5, dy*0.5
        self.dx = dx
        self.dy = self.dx

        # B grid
        B_loc = point(N+1,N+1,ng)
        B_geo = point(N+1,N+1,ng)

        # A grid
        A_loc = point(N,N,ng)
        A_geo = point(N,N,ng)

        # C grid
        C_loc = point(N+1,N,ng)
        C_geo = point(N+1,N,ng)

        # D grid
        D_loc = point(N,N+1,ng)
        D_geo = point(N,N+1,ng)

        #---------------------------------------------------------------------------------
        # Compute B grid
        B_loc.x = np.linspace(-a_ref-ng*dx,a_ref+ng*dx, N+1+2*ng)
        B_loc.y = np.linspace(-a_ref-ng*dy,a_ref+ng*dy, N+1+2*ng)

        if gtype =='equiedge':
            ie = self.ie
            R_ref = self.R_ref
            #print(B_loc.x[3])
            B_loc.x[2] = np.arctan(np.tan(-np.pi*0.5 - np.arctan(R_ref*np.tan(B_loc.x[4])))/R_ref)
            B_loc.x[1] = np.arctan(np.tan(-np.pi*0.5 - np.arctan(R_ref*np.tan(B_loc.x[5])))/R_ref)
            B_loc.x[0] = np.arctan(np.tan(-np.pi*0.5 - np.arctan(R_ref*np.tan(B_loc.x[6])))/R_ref)
            B_loc.x[ie+1] = -B_loc.x[2]
            B_loc.x[ie+2] = -B_loc.x[1]
            B_loc.x[ie+3] = -B_loc.x[0]
            B_loc.y[:] = B_loc.x[:]
        for i in range(self.jsd,self.jed+1):
            print(i-2,B_loc.x[i])
        exit()
 
        B_loc.X, B_loc.Y, B_loc.Z, B_loc.lon, B_loc.lat = \
        equiangular_gnomonic_map(B_loc.x, B_loc.y, N+1, N+1, ng, self.R_ref)

        B_geo.X, B_geo.Y, B_geo.Z, B_geo.lon, B_geo.lat = \
        B_loc.X, B_loc.Y, B_loc.Z, B_loc.lon, B_loc.lat
        #---------------------------------------------------------------------------------



        #---------------------------------------------------------------------------------
        # Compute A grid
        # using local coordinates
        A_loc.x = np.linspace(-a_ref-ng*dx+dxo2,a_ref+ng*dx-dxo2, N+2*ng)
        A_loc.y = np.linspace(-a_ref-ng*dy+dyo2,a_ref+ng*dy-dyo2, N+2*ng)
        A_loc.X, A_loc.Y, A_loc.Z, A_loc.lon, A_loc.lat = \
        equiangular_gnomonic_map(A_loc.x, A_loc.y, N, N, ng, self.R_ref)

        # using geodesic midpoints
        A_geo.X[:,:,:] = (B_geo.X[1:,1:,:]+B_geo.X[1:,:Nt,:]+B_geo.X[:Nt,1:,:]+B_geo.X[:Nt,:Nt,:])
        A_geo.Y[:,:,:] = (B_geo.Y[1:,1:,:]+B_geo.Y[1:,:Nt,:]+B_geo.Y[:Nt,1:,:]+B_geo.Y[:Nt,:Nt,:])
        A_geo.Z[:,:,:] = (B_geo.Z[1:,1:,:]+B_geo.Z[1:,:Nt,:]+B_geo.Z[:Nt,1:,:]+B_geo.Z[:Nt,:Nt,:])
        norm = np.sqrt(A_geo.X*A_geo.X + A_geo.Y*A_geo.Y + A_geo.Z*A_geo.Z)
        A_geo.X, A_geo.Y, A_geo.Z = A_geo.X/norm, A_geo.Y/norm, A_geo.Z/norm
        A_geo.lon, A_geo.lat = cart2sph(A_geo.X, A_geo.Y, A_geo.Z)
        #---------------------------------------------------------------------------------


        #---------------------------------------------------------------------------------
        # Compute C grid
        # using local coordinates
        C_loc.x = np.linspace(-a_ref-ng*dx,a_ref+ng*dx, N+1+2*ng)
        C_loc.y = np.linspace(-a_ref-ng*dy+dyo2,a_ref+ng*dy-dyo2, N+2*ng)
        C_loc.X, C_loc.Y, C_loc.Z, C_loc.lon, C_loc.lat = \
        equiangular_gnomonic_map(C_loc.x, C_loc.y, N+1, N, ng, self.R_ref)

        # using geodesic midpoints
        C_geo.X[:,:,:] = (B_geo.X[:,1:,:]+B_geo.X[:,:Nt,:])
        C_geo.Y[:,:,:] = (B_geo.Y[:,1:,:]+B_geo.Y[:,:Nt,:])
        C_geo.Z[:,:,:] = (B_geo.Z[:,1:,:]+B_geo.Z[:,:Nt,:])
        norm = np.sqrt(C_geo.X*C_geo.X + C_geo.Y*C_geo.Y + C_geo.Z*C_geo.Z)
        C_geo.X, C_geo.Y, C_geo.Z = C_geo.X/norm, C_geo.Y/norm, C_geo.Z/norm
        C_geo.lon, C_geo.lat = cart2sph(C_geo.X, C_geo.Y, C_geo.Z)
        #---------------------------------------------------------------------------------


        #---------------------------------------------------------------------------------
        # Compute D grid
        # using local coordinates
        D_loc.x = np.linspace(-a_ref-ng*dx+dxo2,a_ref+ng*dx-dxo2, N+2*ng)
        D_loc.y = np.linspace(-a_ref-ng*dy,a_ref+ng*dy, N+1+2*ng)
        D_loc.X, D_loc.Y, D_loc.Z, D_loc.lon, D_loc.lat = \
        equiangular_gnomonic_map(D_loc.x, D_loc.y, N, N+1, ng, self.R_ref)

        # using geodesic midpoints
        D_geo.X[:,:,:] = (B_geo.X[1:,:,:]+B_geo.X[:Nt,:,:])
        D_geo.Y[:,:,:] = (B_geo.Y[1:,:,:]+B_geo.Y[:Nt,:,:])
        D_geo.Z[:,:,:] = (B_geo.Z[1:,:,:]+B_geo.Z[:Nt,:,:])
        norm = np.sqrt(D_geo.X*D_geo.X + D_geo.Y*D_geo.Y + D_geo.Z*D_geo.Z)
        D_geo.X, D_geo.Y, D_geo.Z = D_geo.X/norm, D_geo.Y/norm, D_geo.Z/norm
        D_geo.lon, D_geo.lat = cart2sph(D_geo.X, D_geo.Y, D_geo.Z)
        #---------------------------------------------------------------------------------


        #---------------------------------------------------------------------------------
        # Compute areas
        self.areas = metric_tensor(A_loc.x, A_loc.y, self.R_ref)*self.dx*self.dy
        self.ratio_areas = np.amax(self.areas[self.IS:self.ie,self.js:self.je])\
        /np.amin(self.areas[self.IS:self.ie,self.js:self.je])
        #---------------------------------------------------------------------------------

        # store classes
        self.A_loc, self.A_geo = A_loc, A_geo
        self.B_loc, self.B_geo = B_loc, B_geo
        self.C_loc, self.C_geo = C_loc, C_geo
        self.D_loc, self.D_geo = D_loc, D_geo

        self.A_gap = np.sqrt((A_geo.X-A_loc.X)**2 + (A_geo.Y-A_loc.Y)**2 + (A_geo.Z-A_loc.Z)**2)
        self.C_gap = np.sqrt((C_geo.X-C_loc.X)**2 + (C_geo.Y-C_loc.Y)**2 + (C_geo.Z-C_loc.Z)**2)
        self.D_gap = np.sqrt((D_geo.X-D_loc.X)**2 + (D_geo.Y-D_loc.Y)**2 + (D_geo.Z-D_loc.Z)**2)
        #print(np.amax(self.A_gap), np.amax(self.C_gap), np.amax(self.D_gap))

        i0, iend = self.IS, self.ie
        j0, jend = self.js, self.je

####################################################################################
# This routine plots the cubed-sphere grid.
####################################################################################
# Figure format
#fig_format = 'pdf'
fig_format = 'png'
def plot_grid(grid, map_projection):
    # Figure resolution
    dpi = 100

    # Interior cells index (we are ignoring ghost cells)
    IS = grid.IS
    ie = grid.ie
    js = grid.js
    je = grid.je


    isd = grid.isd
    ied = grid.ied
    jsd = grid.jsd
    jed = grid.jed
    # Color of each cubed panel
    #colors = ('blue','red','blue','red','green','green')
    colors = ('black','black','black','black','black','black')
    colors = ('black','white','white','white','white','white')
    print("--------------------------------------------------------")
    print('Plotting '+grid.name+' cubed-sphere grid using '+map_projection+' projection...')


    if map_projection == "mercator":
        plateCr = ccrs.PlateCarree()
        plt.figure(figsize=(1000/dpi, 1000/dpi), dpi=dpi)
    elif map_projection == 'sphere':
        #plateCr = ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0)
        plateCr = ccrs.Orthographic(central_longitude=-45.0, central_latitude=0.0)
        plt.figure(figsize=(800/dpi, 800/dpi), dpi=dpi)
    else:
        print('ERROR: Invalid map projection.')
        exit()

    plateCr._threshold = plateCr._threshold/10.
    ax = plt.axes(projection=plateCr)
    ax.stock_img()

    for p in range(0, nbfaces):
        # A grid
        lonA = grid.A_loc.lon[:,:,p]*rad2deg
        latA = grid.A_loc.lat[:,:,p]*rad2deg

        # B grid
        lonB = grid.B_loc.lon[:,:,p]*rad2deg
        latB = grid.B_loc.lat[:,:,p]*rad2deg

        # C grid
        lonC = grid.C_loc.lon[:,:,p]*rad2deg
        latC = grid.C_loc.lat[:,:,p]*rad2deg

        # D grid
        lonD = grid.D_loc.lon[:,:,p]*rad2deg
        latD = grid.D_loc.lat[:,:,p]*rad2deg

        # Plot geodesic
        A_lon, A_lat = lonB[IS:ie, js:je], latB[IS:ie, js:je]
        A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)
	
        B_lon, B_lat = lonB[IS+1:ie+1, js:je], latB[IS+1:ie+1, js:je]
        B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)

        C_lon, C_lat = lonB[IS+1:ie+1, js+1:je+1], latB[IS+1:ie+1, js+1:je+1]
        C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

        D_lon, D_lat = lonB[IS:ie, js+1:je+1], latB[IS:ie, js+1:je+1]
        D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

        # lines width
        lw = 0.25
        ax.plot([A_lon, B_lon], [A_lat, B_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
        ax.plot([B_lon, C_lon], [B_lat, C_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
        ax.plot([C_lon, D_lon], [C_lat, D_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
        ax.plot([D_lon, A_lon], [D_lat, A_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
        ax.plot([D_lon, A_lon], [D_lat, A_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
 
        if p==0:
           # Plot geodesic
           A_lon, A_lat = lonB[isd:ied, jsd:jed], latB[isd:ied, jsd:jed]
           A_lon, A_lat = np.ndarray.flatten(A_lon), np.ndarray.flatten(A_lat)
	
           B_lon, B_lat = lonB[isd+1:ied+1, jsd:jed], latB[isd+1:ied+1, jsd:jed]
           B_lon, B_lat = np.ndarray.flatten(B_lon), np.ndarray.flatten(B_lat)

           C_lon, C_lat = lonB[isd+1:ied+1, jsd+1:jed+1], latB[isd+1:ied+1, jsd+1:jed+1]
           C_lon, C_lat = np.ndarray.flatten(C_lon),np.ndarray.flatten(C_lat)

           D_lon, D_lat = lonB[isd:ied, jsd+1:jed+1], latB[isd:ied, jsd+1:jed+1]
           D_lon, D_lat = np.ndarray.flatten(D_lon),np.ndarray.flatten(D_lat)

           # lines width
           lw = 0.25
           ax.plot([A_lon, B_lon], [A_lat, B_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
           ax.plot([B_lon, C_lon], [B_lat, C_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
           ax.plot([C_lon, D_lon], [C_lat, D_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
           ax.plot([D_lon, A_lon], [D_lat, A_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
           ax.plot([D_lon, A_lon], [D_lat, A_lat], '-', linewidth=lw, color=colors[p], transform=ccrs.Geodetic())
        
        if p == 0:
            # Plot B grid
            for i in range(isd, ied+1):
                for j in range(jsd, jed+1):
                    if (i>IS and i<ie) and  (j>js and j<je) :
                       ax.plot(lonB[i,j], latB[i,j], marker='o', markersize=3, color = 'white', transform=ccrs.Geodetic())
                    else:
                        ax.plot(lonB[i,j], latB[i,j], marker='o',color = 'blue', transform=ccrs.Geodetic())
        else:
            # Plot B grid
            for i in range(IS, ie+1):
                for j in range(js, je+1):
                       ax.plot(lonB[i,j], latB[i,j], marker='o', markersize=3, color = 'white', transform=ccrs.Geodetic())

            # Plot A grid
            #for i in range(IS, ie):
            #    for j in range(js, je):
            #        ax.plot(lonA[i,j], latA[i,j], marker='o',color = 'white', transform=ccrs.Geodetic())

            # Plot C grid
            #for i in range(IS, ie+1):
            #    for j in range(js, je):
            #        ax.plot(lonC[i,j], latC[i,j], marker='s',color = 'blue', transform=ccrs.Geodetic())

            # Plot D grid
            #for i in range(IS, ie):
            #   for j in range(js, je+1):
            #        ax.plot(lonD[i,j], latD[i,j], marker='s',color = 'red', transform=ccrs.Geodetic())



    plt.xlim(-90,90)
    # Save the figure
    plt.savefig(graphdir+grid.name+"_"+map_projection+'.'+fig_format, format=fig_format)
    print('Figure has been saved in '+graphdir+grid.name+"_"+map_projection+'.'+fig_format)
    print("--------------------------------------------------------\n")
    plt.close()

def main():
    plotprojection = 'mercator'
    #plotprojection = 'sphere'
    N = 48
    #cs1 = CS(N, 'equiangular')
    #plot_grid(cs1, plotprojection)
    cs1 = CS(N, 'equiedge')
    plot_grid(cs1, plotprojection)

    exit()
    gtypes = ('equiangular', 'equiedge')
    Ns = (48, 96, 192, 384, 768)
    Ns = (48, 96, 192, 384,)
    offset_Agrid = np.zeros((len(Ns), len(gtypes)))
    ratio_areas  = np.zeros((len(Ns), len(gtypes)))

    # plot offsets
    for g in range(0, len(gtypes)):
        gtype = gtypes[g]
        for n in range(0, len(Ns)):
            N = Ns[n]
            cs = CS(N, gtype)
            offset_Agrid[n,g] = np.amax(cs.A_gap)
            ratio_areas[n,g] = cs.ratio_areas

    # plot offsets
    for g in range(0, len(gtypes)):
        gtype = gtypes[g]
        CR = (np.log(offset_Agrid[n-1,g])-np.log(offset_Agrid[n,g]))/np.log(2.0)
        CR = "{:10.2f}".format(CR)
        plt.loglog(Ns, offset_Agrid[:,g], label=gtype+', order '+CR, marker='o')
    plt.xlabel(r'$N$')
    plt.ylabel('Difference between local and geodesic A grid points', fontsize=10)
    plt.legend()
    plt.grid(True, which="both")
    #plt.title(title)
    plt.savefig(graphdir+'offset_Agrid', format='pdf') 
    plt.close()

    # plot offsets
    for g in range(0, len(gtypes)):
        gtype = gtypes[g]
        plt.semilogx(Ns, ratio_areas[:,g], label=gtype, marker='o')
    plt.xlabel(r'$N$')
    plt.ylabel('Ratio max/min area', fontsize=10)
    plt.legend()
    plt.grid(True, which="both")
    #plt.title(title)
    plt.savefig(graphdir+'ratio_area', format='pdf') 
    plt.close()


if __name__ == "__main__":
    main()

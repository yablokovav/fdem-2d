import numpy as np
from scipy.sparse import csc_matrix
import time
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from numba import jit, prange, types
from numba import njit
from joblib import Parallel, delayed

#======================================================#
#   setting the arrays for sparse matrix forming for   #
#   Elastic 2D equations                               #
#   Standard Stagered grid with PML and FREE SURFACE   #
#   in frequecncy domain,                              # 
#   code is developed by Aleksander S. Serdyukov       # 
#   aleksanderserdyukov@ya.ru                     
#   code is modified by Alexandr Yablokov
#   yablokovav@ipgg.sbras.ru
#======================================================#

def solve_multi_frequency_elastic(frequencies, vp, vs, qp, qs, rho, nx, nz, h, npml, souy, soux, ampl_sou, verbose=True):
    """
    Solve elastic wave equation for multiple frequencies using parallel processing.
    
    This function computes the elastic wave field solutions for a range of frequencies
    by parallelizing the finite-difference elastic solver across multiple CPU cores.
    
    Parameters:
    -----------
    frequencies : array_like
        List of frequencies (in Hz) to simulate
    vp : ndarray
        P-wave velocity field (2D array)
    vs : ndarray 
        S-wave velocity field (2D array)
    qp : array_like
        P-wave quality factors for each frequency
    qs : array_like
        S-wave quality factors for each frequency  
    rho : ndarray
        Density field (2D array)
    nx : int
        Number of grid points in x-direction
    nz : int
        Number of grid points in z-direction
    h : float
        Grid spacing (in meters)
    npml : int
        Number of PML (Perfectly Matched Layer) boundary points
    souy : float
        Source y-coordinate (depth position)
    soux : float
        Source x-coordinate (lateral position)
    ampl_sou : float
        Source amplitude
    verbose : bool, optional
        Whether to print timing information (default: True)
    
    Returns:
    --------
    data_Yy : ndarray
        Y-component (vertical) wavefield data for all frequencies
    data_Yx : ndarray  
        X-component (horizontal) wavefield data for all frequencies
    """
    
    # Start timing the execution
    start_time = time.perf_counter() 

    # Parallel processing: run elastic solver for each frequency
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(run_fd_elastic_solver)(vp, vs, qp[i], qs[i], rho, nx, nz, h, npml, f, souy, soux, ampl_sou)
        for i, f in enumerate(frequencies)
    )
    
    # Unzip results into separate arrays for Y and X components
    data_Yy, data_Yx = zip(*results)
    
    # Print execution time if verbose mode is enabled
    if verbose:
        elapsed = time.perf_counter() - start_time
        mins, secs = divmod(elapsed, 60)
        print(f"Time elapsed: {elapsed:.1f}s ({mins:.0f}m {secs:.1f}s)")
    
    # Convert results to numpy arrays for easier manipulation
    return np.array(data_Yy), np.array(data_Yx)


def run_fd_elastic_solver(Vp_, Vs_, qp_, qs_, rho_, nx, nz, h, n_pml, f, souy, soux, ampl_sou):
    """
    Run finite-difference elastic wave solver for a single frequency.
    
    This function sets up and solves the elastic wave equation using finite-difference
    methods with complex-valued velocities to account for attenuation.
    
    Parameters:
    -----------
    Vp_ : ndarray
        P-wave velocity field (2D array)
    Vs_ : ndarray
        S-wave velocity field (2D array)
    qp_ : float
        P-wave quality factor for current frequency
    qs_ : float
        S-wave quality factor for current frequency
    rho_ : ndarray
        Density field (2D array)
    nx : int
        Number of grid points in x-direction
    nz : int
        Number of grid points in z-direction
    h : float
        Grid spacing (in meters)
    n_pml : int
        Number of PML boundary points
    f : float
        Current frequency (in Hz)
    souy : float
        Source y-coordinate (depth position)
    soux : float
        Source x-coordinate (lateral position)
    ampl_sou : float
        Source amplitude
    
    Returns:
    --------
    data_Yy : ndarray
        Y-component (vertical) wavefield data for current frequency
    data_Yx : ndarray
        X-component (horizontal) wavefield data for current frequency
    """
    
    # Apply attenuation by making velocities complex-valued
    # The imaginary part represents energy loss (attenuation)
    Vp_ = Vp_ * (1 - 1j / (2 * qp_))
    Vs_ = Vs_ * (1 - 1j / (2 * qs_))
    
    # Generate finite-difference system matrix for elastic wave equation
    # S: matrix values, I, J: row and column indices
    S, I, J = fd_elastic_solver(Vp_, Vs_, rho_, nx, nz, h, n_pml, 2 * np.pi * f)
    
    # Create sparse matrix in Compressed Sparse Column format for efficient solving
    Sm = csc_matrix((S, (I, J)), shape=(2 * nx * nz - nx - nz + 1, 2 * nx * nz - nx - nz + 1), dtype=complex)
    
    # Factorize the matrix for efficient solving (LU decomposition)
    B = splu(Sm)
    
    # Convert source coordinates to grid indices
    n0 = int(souy / h)  # Depth index
    m0 = int(soux / h)   # Lateral index
    
    # Calculate indices for Y and X components in the solution vector
    # The solution vector contains both Y and X components interleaved
    idx_y = np.arange(nx - 1)[:, None] * (2 * nz - 1) + 2 * np.arange(nz - 1) + 1
    idx_x = np.arange(nx - 1)[:, None] * (2 * nz - 1) + 2 * np.arange(nz - 1)
    
    # Create source vector (right-hand side of the equation)
    # Place source at the correct position in the vector
    D = np.zeros(2 * nx * nz - nx - nz + 1, dtype=complex)
    D[2 * n0 + 1 + m0 * (2 * nz - 1)] = ampl_sou / rho_[n0, m0]
    
    # Solve the linear system: Sm * U = D
    U = B.solve(D)

    # Extract Y and X components from the solution vector
    data_Yy = U[idx_y].T  # Transpose to match expected shape
    data_Yx = U[idx_x].T

    return data_Yy, data_Yx

@jit(nopython=True, cache=True)
def fd_elastic_solver(Vp_, Vs_, rho_, nx, nz, h, n_pml, omega):    

    h = 0.5*h
    nz_ = nz - n_pml
    nx_ = nx - 2*n_pml
    
    NX_ = 2*nx_+1
    NZ_ = 2*nz_+1
    N_pml = 2*n_pml
    
    l_pml = h*N_pml  
    NX = NX_ + 2*N_pml
    NZ = NZ_ + N_pml
    
    Vp  = np.zeros((NZ,NX), dtype=np.complex128)
    Vs  = np.zeros((NZ,NX), dtype=np.complex128)
    Rho = np.zeros((NZ,NX))

    for i in range(nz):
        for j in range(nx):
            Vp[i*2,j*2] = Vp_[i,j]
            Vp[i*2+1,j*2+1] = Vp_[i,j]
            Vp[i*2+1,j*2] = Vp_[i,j]
            Vp[i*2,j*2+1] = Vp_[i,j]

            Vs[i*2,j*2] = Vs_[i,j]
            Vs[i*2+1,j*2+1] = Vs_[i,j]
            Vs[i*2+1,j*2] = Vs_[i,j]
            Vs[i*2,j*2+1] = Vs_[i,j]

            Rho[i*2,j*2] = rho_[i,j]
            Rho[i*2+1,j*2+1] = rho_[i,j]
            Rho[i*2+1,j*2] = rho_[i,j]
            Rho[i*2,j*2+1] = rho_[i,j]
        
    Vp[NZ-1,:] = Vp[NZ-2,:]
    Vp[:,NX-1] = Vp[:,NX-2]
    Vs[NZ-1,:] = Vs[NZ-2,:]
    Vs[:,NX-1] = Vs[:,NX-2]
    Rho[NZ-1,:] = Rho[NZ-2,:]
    Rho[:,NX-1] = Rho[:,NX-2]    
        #PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML
    Sx = np.zeros((NZ,NX), dtype=types.complex128)
    Sz = np.zeros((NZ,NX), dtype=types.complex128)
    
    for m in range( N_pml, NX_ + N_pml):
        for n in range(NZ):
            Sx[n,m] = 1.0;
    
    for m in range(N_pml):
        for n in range(NZ):
            lx = (N_pml-m) * h
            l = lx
            Sx[n,m] = 1.0/(1-1j*2.0 *1.79*np.pi*(l/l_pml)**2) 
    
    for m in range(NX_+N_pml,NX):
        for n in range(NZ):
            lx = (m - (NX_+N_pml) + 1)*h 
            l = lx
            Sx[n,m] = 1.0/(1-1j*2.0 *1.79*np.pi*(l/l_pml)**2)
    
    #plt.plot(np.real(Sx[1,:]))
    
    for m in range(NX):
        for n in range(NZ_):
            Sz[n,m] = 1.0;    
    
    for m in range(NX):
        for n in range(NZ_,NZ):
            lz = (n - NZ_ + 1)*h 
            l = lz
            Sz[n,m] = 1.0/(1-1j*2.0 *1.79*np.pi*(l/l_pml)**2)
    
    #PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML_PML

    rho = np.zeros((NX*NZ))
    lamda = np.zeros((NX*NZ), dtype=np.complex128)
    mu = np.zeros((NX*NZ), dtype=np.complex128)
    
    for m in range(NX):
        for n in range(NZ):
            rho[n+m*NZ] = Rho[n,m]
            mu[n+m*NZ] = Rho[n,m]*Vs[n,m]*Vs[n,m]
            lamda[n+m*NZ]= Rho[n,m]*(Vp[n,m]*Vp[n,m] - 2*Vs[n,m]*Vs[n,m])
            
    kk = 0
    #MATRIX_MATRIX_MATRIX_MATRIX_MATRIX_MATRIX
    nzmax = (nz-2)*(nx-3)*9 + (nz-3)*(nx-3)*9 + (nz-2)*9 + (nz-2)*6 + (nz-3)*8 + (nz-2)*6 + (nz-3)*8 + (nx-3)*6 + (nx-3)*8 + 12 + (nx-3)*14 + 44
    s = np.zeros(nzmax, dtype=types.complex128)  # value nonzero elements sparse-matrix
    i = np.zeros(nzmax, dtype=types.int64)  # row sparse-matrix
    j = np.zeros(nzmax, dtype=types.int64)  # column sparse-matrix
    
    #CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL__CENTRAL
    for n2 in range(1,nz-1):
        for m2 in range(1,nx-2):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2------здесь индексы на полной сетке
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j--------------------  _I
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1        
            #5----Vx_i,j+2------------------  _I
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #8----Vx_i-2,j------------------- _I
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) + m2*(2*nz-1))
            kk +=1
            #9----Vx_i+2,j------------------- _I
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #12----Vz_i-1,j-1----------------_I
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*mu[n-1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1+ (m2-1)*(2*nz-1))
            kk +=1
            #10----Vz_i+1,j-1---------------- _I
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #11----Vz_i+1,j+1---------------- _I
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n+1,m]*Sx[n+1,m+1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1
            #13----Vz_i-1,j+1---------------- _I
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sx[n-1,m+1]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            
    for n2 in range(1,nz-2):
        for m2 in range(1,nx-2):  
            n = n2*2 + 2
            m = m2*2 + 2
            
    #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II
            #12----Vx_i-1,j-1---------------_II
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1
            #10----Vx_i+1,j-1---------------_II
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1--------------_II
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1--------------_II
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) -  Sz[n-1,m]*Sx[n-1,m+1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1))
            kk +=1
    #========================================================================================================        
            #1-----Vz_i,j-2--------------_II
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ])) 
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #3-----Vz_i,j----------------_II
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1        
            #5-----Vz_i,j+2------------- _II
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2+1)*(2*nz-1))
            kk +=1
            #8-----Vz_i-2,j------------- _II
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            #9----Vz_i+2,j-------------- _II
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) +1 + m2*(2*nz-1))
            kk +=1   

#CENT_X_CENT_X_CENT_X_CENT_X_CENT_X_CENT_X_CENT_X_CENT_X_CENT_X_CENT_X_
    for n2 in range(1,nz-1):
        for m2 in range(nx-2,nx-1):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2------CENT_X_
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j--------------  CENT_X_ _I
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1        
            #5----Vx_i,j+2-------------CENT_X_ _I
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #8----Vx_i-2,j--------------CENT_X_ _I
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) + m2*(2*nz-1))
            kk +=1
            #9----Vx_i+2,j--------------CENT_X_ _I
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #12----Vz_i-1,j-1-----------CENT_X_ _I
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*mu[n-1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1+ (m2-1)*(2*nz-1))
            kk +=1
            #10----Vz_i+1,j-1-----------CENT_X_ _I
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #11----Vz_i+1,j+1-----------CENT_X_ _I
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n+1,m]*Sx[n+1,m+1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1
            #13----Vz_i-1,j+1-------------CENT_X_ _I
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sx[n-1,m+1]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
               
    #LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT__LEFT
    for n2 in range(1,nz-1):
        for m2 in range(1):
            n = n2*2+1
            m = m2*2+1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I             
            #3----Vx_i,j------------------_I LEFT
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1        
            #5----Vx_i,j+2----------------_I LEFT
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #8----Vx_i-2,j----------------_I LEFT
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) + m2*(2*nz-1))
            kk +=1
            #9----Vx_i+2,j----------------_I LEFT
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #11----Vz_i+1,j+1-------------_I LEFT
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n+1,m]*Sx[n+1,m+1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1
            #13----Vz_i-1,j+1-------------_I LEFT
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sx[n-1,m+1]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            
    for n2 in range(1,nz-2):
        for m2 in range(1):
            n = n2*2 + 2
            m = m2*2 + 2
    #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II
            #12----Vx_i-1,j-1------------- _II LEFT
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1
            #10----Vx_i+1,j-1------------- _II LEFT
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1------------  _II LEFT
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1------------  _II LEFT
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) -  Sz[n-1,m]*Sx[n-1,m+1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1))
            kk +=1
    #========================================================================================================         
            #3-----Vz_i,j--------------_II LEFT
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1        
            #5-----Vz_i,j+2------------_II LEFT
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2+1)*(2*nz-1))
            kk +=1
            #8-----Vz_i-2,j------------_II LEFT
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            #9----Vz_i+2,j------------_II LEFT
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) +1 + m2*(2*nz-1))
            kk +=1
            
    #RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT_RIGHT
    for n2 in range(1,nz-1):
        for m2 in range(nx-1,nx):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2----------------_I RIGHT
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j------------------_I RIGHT
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(n2 + m2*(2*nz-1))
            kk +=1           
            #8----Vx_i-2,j----------------_I RIGHT
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=((n2-1) + m2*(2*nz-1))
            kk +=1
            #9----Vx_i+2,j----------------_I RIGHT
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=((n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #12----Vz_i-1,j-1-------------_I RIGHT
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*mu[n-1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1+ (m2-1)*(2*nz-1))
            kk +=1
            #10----Vz_i+1,j-1-------------_I RIGHT
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            
    for n2 in range(1,nz-2):
        for m2 in range(nx-2,nx-1):
            n = n2*2 + 2
            m = m2*2 + 2            
    #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II
            #12----Vx_i-1,j-1-------------_II RIGHT
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1
            #10----Vx_i+1,j-1-------------_II RIGHT
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1------------_II RIGHT
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=((n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1------------_II RIGHT
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) -  Sz[n-1,m]*Sx[n-1,m+1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(n2 + (m2+1)*(2*nz-1))
            kk +=1
    #========================================================================================================        
            #1-----Vz_i,j-2--------------_II RIGHT
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ])) 
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #3-----Vz_i,j----------------_II RIGHT
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1                
            #8-----Vz_i-2,j--------------_II RIGHT
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            #9----Vz_i+2,j---------------_II RIGHT
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) +1 + m2*(2*nz-1))
            kk +=1
    
    #UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_UP_FS!_FS!_FS!_FS!_FS!_FS!_FS!_FS!_FS!
    # FS! = FREE SURF
    for n2 in range(1):
        for m2 in range(1,nx-2):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I  _FS!_FS!_FS!_FS!     
            #1----Vx_i,j-2-------------------_I UP _FS!
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+mu[n+(m-1)*NZ])*mu[n+(m-1)*NZ]/((lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j---------------------_I UP  _FS!
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+mu[n+(m-1)*NZ])*mu[n+(m-1)*NZ]/((lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+mu[n+(m+1)*NZ])*mu[n+(m+1)*NZ]/((lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])*h*h*rho[n+m*NZ]) - 2*Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1       
            #5----Vx_i,j+2-------------------_I UP  _FS!
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+mu[n+(m+1)*NZ])*mu[n+(m+1)*NZ]/((lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #9----Vx_i+2,j-------------------_I UP _FS!
            s[kk]=(2*Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================

            #10----Vz_i+1,j-1---------------_I UP  _FS!
            s[kk]=(-2*Sz[n+1,m]*Sx[n+1,m-1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #11----Vz_i+1,j+1---------------_I UP _FS!
            s[kk]=(2*Sz[n+1,m]*Sx[n+1,m+1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1
            
    for n2 in range(1):
        for m2 in range(1,nx-2):
            n = n2*2 + 2
            m = m2*2 + 2
    #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_FS!FS!FS!FS!FS!FS!FS!
            #12----Vx_i-1,j-1---------------_II UP __FS!
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1            
            #10----Vx_i+1,j-1---------------_II UP___FS!
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1---------------_II UP___FS!
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1---------------_II UP __FS!
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1))
            kk +=1
    #========================================================================================================        
            #1-----Vz_i,j-2----------------_II UP___FS!
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ])) 
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #3-----Vz_i,j------------------_II UP___FS!
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1        
            #5-----Vz_i,j+2----------------_II UP___FS!
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2+1)*(2*nz-1))
            kk +=1
            #9----Vz_i+2,j-----------------_II UP___FS!
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) +1 + m2*(2*nz-1))
            kk +=1

#UP_X__UP_X__UP_X__UP_X__UP_X__UP_X__UP_X___FS!_FS!_FS!_FS!_FS!_FS!_FS!_FS!
    for n2 in range(1):
        for m2 in range(nx-2,nx-1):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2-------------------_I UP_X _FS!
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+mu[n+(m-1)*NZ])*mu[n+(m-1)*NZ]/((lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j---------------------_I UP_X  _FS!
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+mu[n+(m-1)*NZ])*mu[n+(m-1)*NZ]/((lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+mu[n+(m+1)*NZ])*mu[n+(m+1)*NZ]/((lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])*h*h*rho[n+m*NZ]) - 2*Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1       
            #5----Vx_i,j+2-------------------_I UP_X  _FS!
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+mu[n+(m+1)*NZ])*mu[n+(m+1)*NZ]/((lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!         
            j[kk]=(n2 + (m2+1)*(2*nz-1)) 
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!         
            kk +=1
            #9----Vx_i+2,j-------------------_I UP_X _FS!
            s[kk]=(2*Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #10----Vz_i+1,j-1---------------_I UP_X _FS!
            s[kk]=(- 2*Sz[n+1,m]*Sx[n+1,m-1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #11----Vz_i+1,j+1---------------_I UP_X _FS!
            s[kk]=(2*Sz[n+1,m]*Sx[n+1,m+1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1              
    #DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN_DOWN
    for n2 in range(nz-1,nz):
        for m2 in range(1,nx-2):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2--------------------_I DOWN
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j----------------------_I DOWN
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1       
            #5----Vx_i,j+2---------------------_I DOWN
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #8----Vx_i-2,j----------------------_I DOWN 
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #12----Vz_i-1,j-1----------------_I DOWN 
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*mu[n-1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1+ (m2-1)*(2*nz-1))
            kk +=1
            #13----Vz_i-1,j+1----------------_I DOWN 
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sx[n-1,m+1]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            
    for n2 in range(nz-2,nz-1):
        for m2 in range(1,nx-2):  
            n = n2*2 + 2
            m = m2*2 + 2
            
    #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II
            #12----Vx_i-1,j-1---------------_II DOWN
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1
            #10----Vx_i+1,j-1---------------_II DOWN
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1---------------_II DOWN
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1---------------_II DOWN
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) -  Sz[n-1,m]*Sx[n-1,m+1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1))
            kk +=1
    #========================================================================================================        
            #1-----Vz_i,j-2------------------_II DOWN
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ])) 
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #3-----Vz_i,j--------------------_II DOWN
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1           
            #5-----Vz_i,j+2------------------_II DOWN
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2+1)*(2*nz-1))
            kk +=1
            #8-----Vz_i-2,j------------------_II DOWN
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1

#DOWN_X__DOWN_X__DOWN_X__DOWN_X__DOWN_X__DOWN_X__DOWN_X__DOWN_X__DOWN_X__
    for n2 in range(nz-1,nz):
        for m2 in range(nx-2,nx-1):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2--------------------_I DOWN_X__
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j----------------------_I DOWN_X__
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1       
            #5----Vx_i,j+2---------------------_I DOWN_X__
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #8----Vx_i-2,j----------------------_I DOWN_X__ 
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #12----Vz_i-1,j-1----------------_I DOWN_X__ 
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*mu[n-1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1+ (m2-1)*(2*nz-1))
            kk +=1
            #13----Vz_i-1,j+1----------------_I DOWN_X__ 
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sx[n-1,m+1]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1    

##-CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0--CS0-FS!_FS!_FS!_FS!_FS!_FS!_FS!_FS!_
# в углах 
    for n2 in range(1):
        for m2 in range(1):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #3----Vx_i,j--------------------_I CS0 _FS!_
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+mu[n+(m-1)*NZ])*mu[n+(m-1)*NZ]/((lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+mu[n+(m+1)*NZ])*mu[n+(m+1)*NZ]/((lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])*h*h*rho[n+m*NZ]) - 2*Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1        
            #5----Vx_i,j+2------------------_I CS0 _FS!_
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+mu[n+(m+1)*NZ])*mu[n+(m+1)*NZ]/((lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #9----Vx_i+2,j------------------_I CS0 _FS!_
            s[kk]=(2*Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #11----Vz_i+1,j+1---------------_I CS0 _FS!_
            s[kk]=(2*Sz[n+1,m]*Sx[n+1,m+1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1
            
    for n2 in range(1):
        for m2 in range(1):
            n = n2*2 + 2
            m = m2*2 + 2                        
    #II_II_II_II_II_II_II_II_II_II_II_FS!_CS0_FS!_CS0_FS!_CS0_FS!_CS0_FS!
            #12----Vx_i-1,j-1--------------_II  CS0 _FS!
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1
            #10----Vx_i+1,j-1--------------_II  CS0 _FS!
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1--------------_II  CS0 _FS!
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1--------------_II  CS0 _FS!
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1))
            kk +=1            
#========================================================================================================        
            #3-----Vz_i,j------------------_II  CS0 _FS!
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1        
            #5-----Vz_i,j+2----------------_II  CS0_FS!
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2+1)*(2*nz-1))
            kk +=1
            #9----Vz_i+2,j-----------------_II  CS0_FS!
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) +1 + m2*(2*nz-1))
            kk +=1 
            
##-CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1--CS1__FS!_FS!_FS!_FS!_FS!_FS!_FS!_FS!
    for n2 in range(1):
        for m2 in range(nx-1,nx):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2----------------_I  CS1 _FS!_
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+mu[n+(m-1)*NZ])*mu[n+(m-1)*NZ]/((lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j------------------_I CS1 _FS!_
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+mu[n+(m-1)*NZ])*mu[n+(m-1)*NZ]/((lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+mu[n+(m+1)*NZ])*mu[n+(m+1)*NZ]/((lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])*h*h*rho[n+m*NZ]) - 2*Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(n2 + m2*(2*nz-1))
            kk +=1         
            #9----Vx_i+2,j----------------_I CS1 _FS!_
            s[kk]=(2*Sz[n+1,m]*Sz[n+2,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=((n2+1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #10----Vz_i+1,j-1-------------_I CS1 _FS!_
            s[kk]=(- 2*Sz[n+1,m]*Sx[n+1,m-1]*mu[n+1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            
    for n2 in range(1):
        for m2 in range(nx-2,nx-1):        #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_CS1_FS!_CS1_FS!_CS1_FS!
            n = n2*2 + 2
            m = m2*2 + 2
            
            #12----Vx_i-1,j-1--------------_II CS1
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1            
            #10----Vx_i+1,j-1--------------_II CS1_FS!
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1--------------_II CS1_FS!
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=((n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1--------------_II CS1
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(n2 + (m2+1)*(2*nz-1))
            kk +=1#========================================================================================================        
            #1-----Vz_i,j-2--------------_II CS1_FS!
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ])) 
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #3-----Vz_i,j----------------_II CS1_FS!
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1          
            #9----Vz_i+2,j---------------_II CS1_FS!
            s[kk]=(Sz[n+1,m]*Sz[n+2,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) +1 + m2*(2*nz-1))
            kk +=1
            
##-CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2--CS2
    for n2 in range(nz-1,nz):
        for m2 in range(1):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #3----Vx_i,j--------------------_I  CS2
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1          
            #5----Vx_i,j+2------------------_I  CS2
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1)) 
            kk +=1
            #8----Vx_i-2,j------------------_I  CS2
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #13----Vz_i-1,j+1---------------_I  CS2
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*lamda[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sx[n-1,m+1]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            
    for n2 in range(nz-2,nz-1):
        for m2 in range(1):
            n = n2*2 + 2
            m = m2*2 + 2
    #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II
            #12----Vx_i-1,j-1---------------_II  CS2
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1
            #10----Vx_i+1,j-1---------------_II  CS2
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1 
            #11----Vx_i+1,j+1---------------_II  CS2
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1---------------_II  CS2
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) -  Sz[n-1,m]*Sx[n-1,m+1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2+1)*(2*nz-1))
            kk +=1
    #========================================================================================================        
            #3-----Vz_i,j-----------------_II CS2
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1        
            #5-----Vz_i,j+2---------------_II CS2
            s[kk]=(Sx[n,m+1]*Sx[n,m+2]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2+1)*(2*nz-1))
            kk +=1
            #8-----Vz_i-2,j--------------_II CS2
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
            
##-CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3--CS3
    for n2 in range(nz-1,nz):
        for m2 in range(nx-1,nx):
            n = n2*2 + 1
            m = m2*2 + 1
    #I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I_I       
            #1----Vx_i,j-2------------------_I   CS3
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(2*n2 + (m2-1)*(2*nz-1))            
            kk +=1
            #3----Vx_i,j--------------------_I   CS3
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*(lamda[n+(m-1)*NZ]+2*mu[n+(m-1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*(lamda[n+(m+1)*NZ]+2*mu[n+(m+1)*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*mu[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(n2 + m2*(2*nz-1))
            kk +=1         
            #8----Vx_i-2,j------------------_I  CS3
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*mu[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=((n2-1) + m2*(2*nz-1))
            kk +=1
    #============================================================================================================
            #12----Vz_i-1,j-1---------------_I  CS3
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*lamda[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*mu[n-1 + m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(n2 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1+ (m2-1)*(2*nz-1))
            kk +=1
            
    for n2 in range(nz-2,nz-1):
        for m2 in range(nx-2,nx-1):
            n = n2*2 + 2
            m = m2*2 + 2            
    #II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II_II
            #12----Vx_i-1,j-1--------------_II CS3
            s[kk]=(Sx[n,m-1]*Sz[n-1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) + Sz[n-1,m]*Sx[n-1,m-1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*n2 + m2*(2*nz-1))
            kk +=1
            #10----Vx_i+1,j-1--------------_II CS3
            s[kk]=(-Sx[n,m-1]*Sz[n+1,m-1]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sx[n+1,m-1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(2*(n2+1) + m2*(2*nz-1))
            kk +=1
            #11----Vx_i+1,j+1--------------_II CS3
            s[kk]=(Sx[n,m+1]*Sz[n+1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ])+ Sz[n+1,m]*Sx[n+1,m+1]*lamda[n+1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=((n2+1) + (m2+1)*(2*nz-1))
            kk +=1
            #13----Vx_i-1,j+1--------------_II CS3
            s[kk]=(-Sx[n,m+1]*Sz[n-1,m+1]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) -  Sz[n-1,m]*Sx[n-1,m+1]*lamda[n-1+m*NZ]/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 + 1 + m2*(2*nz-1))
            j[kk]=(n2 + (m2+1)*(2*nz-1))
            kk +=1
    #========================================================================================================        
            #1-----Vz_i,j-2------------------_II  CS3
            s[kk]=(Sx[n,m-1]*Sx[n,m-2]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ])) 
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + (m2-1)*(2*nz-1))
            kk +=1
            #3-----Vz_i,j--------------------_II  CS3
            s[kk]=(omega**2 - Sx[n,m-1]*Sx[n,m]*mu[n+(m-1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sx[n,m+1]*Sx[n,m]*mu[n+(m+1)*NZ]/(4*h*h*rho[n+m*NZ]) - Sz[n-1,m]*Sz[n,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]) - Sz[n+1,m]*Sz[n,m]*(lamda[n+1+m*NZ]+2*mu[n+1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*n2 +1 + m2*(2*nz-1))
            kk +=1           
            #8-----Vz_i-2,j-----------------_II CS3
            s[kk]=(Sz[n-1,m]*Sz[n-2,m]*(lamda[n-1+m*NZ]+2*mu[n-1+m*NZ])/(4*h*h*rho[n+m*NZ]))
            i[kk]=(2*n2 +1 + m2*(2*nz-1))
            j[kk]=(2*(n2-1) +1 + m2*(2*nz-1))
            kk +=1
    
    return s, i, j



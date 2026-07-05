import pandas as pd
import numpy as np
from numpy.fft import rfft, irfft
from numpy import pi, cos, linspace, arange
from scipy.fftpack import diff as psdiff
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from scipy.optimize import brentq

# ================================ Functions ===============================
def initialise_ionosphere_region(df, region):
    """ takes a dataframe of ionospheric properties and pulls the associated properties for a given region. making the necessary calculations required to run simulation CURRENTLY NOT ABLE TO HANDLE GEO"""

    # Pulls property name and values into a list and numpy array respectively
    value = df_ionosphere[region].to_numpy()
    properties =  df_ionosphere["Properties"].to_list()

    # Creates a dictionary of the properties and values for the given region
    plasma_properties = dict(zip(properties, value))
    return plasma_properties

def normalised_plasma_potential(a, debye_length, mass_ratio, potential_radius = None):
    """Calculates the normalised plasma potential for a given debris radius, debye length, and mass ratio."""
    def eqn_surface_potential(x, mass_ratio, s, a):
        """Equation 32 from the NASA paper in the form 0 = f(x) where f(x) is equal to g(x) - A(1-B * e(Cx), this equation is then put through brents method to identify the roots of the function, eφ/kT"""
        return np.exp(x) - np.sqrt(mass_ratio) * (s/a)**2  * ( 1 - ((s**2 - a**2)/s**2) * np.exp( (a**2 / (s**2 - a**2)) * x ))

    # If potential_radius is not provided, it defaults to the debye_length.
    if potential_radius is None:
        potential_radius = debye_length

    s = a + debye_length

    x_sol = brentq(eqn_surface_potential, -10, 0, args=(mass_ratio, s, a))
    phi_np = x_sol * np.exp(-potential_radius/debye_length)  # normalised plasma potential

    return phi_np

def forcing_term(np_potential, xi, tau, vd, G):
    return np_potential * np.exp(-1 * ((xi - (vd * tau))/ G)**2)

def fkdv_exact_solution(a, x, K = 0.6124):
    return a / (np.cosh(K * x)**2)

def ck_solutions(u0, time_array, alpha, beta, s, k, f, dt, N):
    """Use the Crank-Nicolson scheme to solve the normalised forced KdV equation on a periodic domain."""

    #initalisng the soltuion array with the initial condition
    sol = np.zeros(((len(time_array)),N))
    sol[1] = u0

    # pre-calculating coefficients
    _ck_beta = dt * beta * 1j * s**3 * k**3
    _ck_z = dt * 1j * s * k

    # computing the rfft of u & f halfs the length of array which which then is multipled by 1024 leng coefficents arrays because of k
    _ck_beta_fft = _ck_beta[:len(k)//2+1]  # Keep only first 513 elements
    _ck_z_fft = _ck_z[:len(k)//2+1]

    # iterate over the time array to compute the solution at each time step
    for i in range(1, len(time_array)-1):
        sol[i+1] = irfft( ((1 - _ck_beta_fft)**-1) * ( (1 + _ck_beta_fft) * rfft(sol[i-1]) - alpha * _ck_z_fft * rfft(sol[i])**2 + _ck_z_fft * rfft(f) ) )

    return sol

# Upload the Ionosphere properties from the Excel file
df_ionosphere= pd.read_excel(r'Ionosphere prop.xltx')
display(df_ionosphere)

# Pulls the properties for the Low LEO region into a dictionary
plasma_prop = initialise_ionosphere_region(df_ionosphere, "Low LEO")
#%%



# =============================== Inputs ================================

# Global constants
__kb__ = 1.380649e-23  #Boltzmann constant in J/K
__c__ = 299792458  # Speed of light in m/s
__me__ = 9.1093837139e-31 # Electron mass in kg
__mh__ = 1.67262192595e-27 # Hydrogen ion aka proton mass in kg
__mo__ = 15.994914619 * 1.66053906892e-27 # Oxygen ion mass in kg

# Physical parameters
a = 0.01 # Debris radius in m
alpha = 1.0051
beta = 0.4925
debris_vel = 0 # Debris velocity in m/s
np_potential = None # Normalised plasma potential, if None, it will be automatically calculated.

# Simulation parameters
N = 1024 # number of discretised points in the domain (power of 2 for FFT speed)
L = 20 # size of the domain
G = a / plasma_prop["Debye Length"] # Normalised forcing width
tau_step = 2 # s
tau_inital = 0 # s
tau_final = 120 # s


# Determine the ion species and set the corresponding ion mass and mass ratio
if plasma_prop["Ion Species"] == "O":
    mi = __mo__
    mass_ratio = __me__ / __mo__
elif plasma_prop["Ion Species"] == "H":
    mi= __mh__
    mass_ratio = __me__ / __mh__
else:
    raise ValueError("Ion species not recognised. Please use 'O' for Oxygen or 'H' for Hydrogen.")

# if up
if np_potential is None:
    np_potential = normalised_plasma_potential(a, plasma_prop["Debye Length"], mass_ratio)


#================================= Main ================================

if __name__ == "__main__":

    # Creating the normalise spae and time arrays
    t = np.linspace(tau_inital, tau_final, N)
    xi = np.linspace(-L, L, N, endpoint=False)
    dxi = xi[1] - xi[0]

    # Defining mesh space & waven numbers
    s = np.pi / L
    # xi =  s * (x + L)
    # dxi = 2 * np.pi / N

    k = np.fft.fftfreq(N, d=1/N)
    print(k.shape)
    # Intialisng with an exact soltuion fot U
    u0 = fkdv_exact_solution(a, xi)
    print(u0.shape, type(u0))

    # Calculating a forcing term for the debris
    np_potential = normalised_plasma_potential(a, plasma_prop["Debye Length"], mass_ratio)
    f = forcing_term(np_potential, xi, t, debris_vel, G)
    print()

    # Propogate exact solution forward in time with CK scheme
    sol = ck_solutions(u0= u0,
                       time_array=t,
                       alpha=alpha,
                       beta=beta,
                       s=s,
                       k=k,
                       f=f,
                       dt=tau_step,
                       N=N
                       )


    print("plotting")
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    ax.scatter(xi, t, sol)
    plt.show()

    plt.figure(figsize=(6,5))
    for i in range(1):
        plt.plot(xi, sol[4])
    plt.xlabel("x")
    plt.ylabel("U")
    plt.grid()
    plt.show()
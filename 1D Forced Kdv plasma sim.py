## Imports & variables
import numpy as np
from numpy.fft import rfft, irfft
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from functools import partial


# ================================ Functions ===============================
def initialise_ionosphere_region(df, region):
    """ takes a dataframe of ionospheric properties and pulls the associated properties for a given region. making the necessary calculations required to run simulation CURRENTLY NOT ABLE TO HANDLE GEO"""

    # Pulls property name and values into a list and numpy array respectively
    value = df[region].to_numpy()
    properties = df["Properties"].to_list()

    # Creates a dictionary of the properties and values for the given region
    plasma_properties = dict(zip(properties, value))
    return plasma_properties

def normalised_plasma_potential(a, debye_length, mass_ratio, potential_radius=None):
    """Calculates normalised plasma potential; ``a`` and ``debye_length`` must use the same length unit."""
    def _eqn_surface_potential(x, mass_ratio, s, a):
        """Equation 32 from the NASA paper in the form 0 = f(x) where f(x) is equal to g(x) - A(1-B * e(Cx), this equation is then put through brents method to identify the roots of the function, eφ/kT"""
        return np.exp(x) - np.sqrt(mass_ratio) * (s / a) ** 2 * (1 - ((s ** 2 - a ** 2) / s ** 2) * np.exp((a ** 2 / (s ** 2 - a ** 2)) * x))

    # If potential_radius is not provided, it defaults to the debye_length.
    if potential_radius is None:
        potential_radius = debye_length

    s = a + debye_length

    x_sol = brentq(_eqn_surface_potential, -10, 0, args=(mass_ratio, s, a))
    phi_np = x_sol * np.exp(-potential_radius / debye_length)  # normalised plasma potential

    return phi_np

def forcing_term_general(np_potential, vd, G, x, time):
    """ Calculates the forcing term for the normalised forced KdV equation
        Parameters: X IS XI AND TIME IS TAU
        Returns: Forcing term as a numpy array
    """
    return np_potential * np.exp(-1 * (((x - vd * time ) / G) ** 2))

def ck_solver(prev_sol, curr_sol, f, k, alpha, beta, dt, s = 1.0):
    """Use the Chan-Kerkhoven scheme to solve the forced KdV equation on a periodic domain."""

    k_fs = k[:len(k) // 2 + 1] # slicng k down 1024 (real) = 1024 // 2 + 1 (fourier space )

    # pre-calculating coefficients
    _ck_beta_fs = dt * beta * 1j * (k_fs ** 3) * (s ** 3)
    _ck_z_fs = dt * 1j * k_fs * s

    # Fourier transform of the last solution, current solution, and forcing term
    fft_prev = rfft(prev_sol)
    fft_sqrd_curr = rfft(curr_sol ** 2)
    fft_f = rfft(f)

    return irfft( ( 1/(1 - _ck_beta_fs) ) * ( (1 + _ck_beta_fs)*fft_prev - alpha*_ck_z_fs*fft_sqrd_curr + _ck_z_fs*fft_f ) )

def fw_solver(prev_sol, curr_sol, f, k, alpha, beta, dt, s = 1.0):
    """Use the Fornberg & Whitham scheme to solve the forced KdV equation on a periodic domain."""

    k_fs = k[:len(k) // 2 + 1]

    fft_f = rfft(f)
    fft_curr = rfft(curr_sol)

    ifft_f_ik = irfft(fft_f * 1j * k_fs)
    ifft_curr_ik = irfft(fft_curr * 1j * k_fs)
    ifft_curr_ik_3 = irfft(fft_curr * (1j ** 3) * (k_fs ** 3))

    return (prev_sol
            - 2 * dt * alpha * curr_sol * s * ifft_curr_ik
            - 2 * dt * beta * (s**3) * ifft_curr_ik_3
            + dt * s * ifft_f_ik)


# =============================== Inputs ================================

df_ionosphere = pd.read_excel(r'Ionosphere prop.xltx') # Upload the Ionosphere properties from the Excel file
plasma_prop = initialise_ionosphere_region(df_ionosphere, "Low LEO") # Pulls the properties for the Low LEO region into a dictionary
display(df_ionosphere)

# Global constants
__kb__ = 1.380649e-23  #Boltzmann constant in J/K
__c__ = 299792458.0  # Speed of light in m/s
__me__ = 9.1093837139e-31  # Electron mass in kg
__mh__ = 1.67262192595e-27  # Hydrogen ion aka proton mass in kg
__mo__ = 15.994914619 * 1.66053906892e-27  # Oxygen ion mass in kg

# Normlaised units
debye_length_m = plasma_prop["Debye Length"] * 1e-2  # spreadsheet value is in cm
vi_km_s = plasma_prop["Ion Acoustic Velocity"]  # spreadsheet value is in km/s
TU = debye_length_m / (vi_km_s * 1e3)  # Time unit in seconds

# ======================================= Inputs ====================================================

# physical parameters
alpha = 1.0051 # Nonlinearity parameter ( 1.0051)
beta = 0.4925 # Dispersion parameter (0.4925)

# forcing parameters
a = 0.01  # debris in m
vd = 0 # Debris velocity in km/s
np_potential = None  # Normalised plasma potential, if None, it will be automatically calculated.
G = a / (plasma_prop["Debye Length"] * 1e-2)  # Normalised forcing width

# Simulation parameters
N = 1024  # number of discretised points in the domain (power of 2 for FFT speed)
L = 100.0  # size of the domain in debye lengths
dtau = 1e-3  # TU
tau_inital = 0.0  # TU
tau_final =  60.0   # TU

# ======================================================================================================

# Determine the ion species and set the corresponding ion mass and mass ratio
if plasma_prop["Ion Species"] == "O":
    mi = __mo__
    mass_ratio = __me__ / __mo__
elif plasma_prop["Ion Species"] == "H":
    mi = __mh__
    mass_ratio = __me__ / __mh__
else:
    raise ValueError("Ion species not recognised. Please use 'O' for Oxygen or 'H' for Hydrogen.")

# if up
if np_potential is None:
    np_potential = normalised_plasma_potential(a, debye_length_m, mass_ratio)
if vd is None:
    vd = plasma_prop["Debris Velocity Circular Orbit"]  # Debris velocity in km/s

vd_norm = vd / vi_km_s  # Normalised debris velocity
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
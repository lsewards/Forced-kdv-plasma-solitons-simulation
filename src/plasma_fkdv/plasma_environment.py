import numpy as np
from scipy.optimize import brentq
from .constants import __me__, __mh__, __mo__

def initialise_ionosphere_region(df, region):
    """ takes a dataframe of ionospheric properties and pulls the associated properties for a given region.
    making the necessary calculations required to run simulation CURRENTLY NOT ABLE TO HANDLE GEO"""

    # Pulls property name and values into a list and numpy array respectively
    value = df[region].to_numpy()
    properties = df["Properties"].to_list()

    return dict(zip(properties, value))   # Creates a dictionary of the properties and values for the given region

def normalized_plasma_potential(a, debye_length, mass_ratio, potential_radius=None):
    """Calculates normalized plasma potential; ``a`` and ``debye_length`` must use the same length unit."""
    def _eqn_surface_potential(x, mass_ratio, s, a):
        """Equation 32 from the NASA paper in the form 0 = f(x) where f(x) is equal to g(x) - A(1-B * e(Cx), this equation is then put through brents method to identify the roots of the function, eφ/kT"""
        return np.exp(x) - np.sqrt(mass_ratio) * (s / a) ** 2 * (1 - ((s ** 2 - a ** 2) / s ** 2) * np.exp((a ** 2 / (s ** 2 - a ** 2)) * x))

    # If potential_radius is not provided, it defaults to the debye_length.
    if potential_radius is None:
        potential_radius = debye_length

    s = a + debye_length
    x_sol = brentq(_eqn_surface_potential, -10, 0, args=(mass_ratio, s, a))

    return  x_sol * np.exp(-potential_radius / debye_length)

def forcing_space_debris(np_potential, vd, G, x, time):
    """ Calculates the forcing term for the normalized forced KdV equation
        Parameters: X IS XI AND TIME IS TAU
        Returns: Forcing term as a numpy array
    """
    return np_potential * np.exp(-1 * (((x - vd * time ) / G) ** 2))

def predict_soliton_type(vd_norm, vi, alpha, beta, np_potential, G):

    A = abs(np_potential) * G * np.sqrt(np.pi)  # area of the gaussian forcing disturbance
    c0 = ( (3 * (alpha/2)**2 * A**2) / 16 * beta ) ** 1/3

    v_sup = (1 + c0) / vi
    v_sub = (1 - c0) / vi

    if vd_norm < v_sub:
        return "Subsonic"
    elif v_sub < vd_norm < v_sup:
        return "Transonic"
    else:
        return "Supersonic"
   
def get_forcing_space_debris_args(plasma_prop, debris_radius, np_potential, vd, G):
    """ Returns a list of pre-defined parameters for the forcing term calculation."""

    # Normalizes plasma properties
    debye_length_m = plasma_prop["Debye Length"] * 1e-2
    vi_km_s = plasma_prop["Ion Acoustic Velocity"]
    TU = debye_length_m / (vi_km_s * 1e3)

    # Unless directly specified, uses default values from spreadsheet or calculation
    if vd is None:
        vd = plasma_prop["Debris Velocity Circular Orbit"]  # Debris velocity in km/s
    if G is None:
        G = debris_radius / debye_length_m  # Normalised forcing width
    if np_potential is None:
        # Determine the ion species and set the corresponding ion mass and mass ratio for the plasma region
        if plasma_prop["Ion Species"] == "O":
            mass_ratio = __me__ / __mo__
        elif plasma_prop["Ion Species"] == "H":
            mass_ratio = __me__ / __mh__
        else:
            raise ValueError("Ion species not recognised. Please use 'O' for Oxygen or 'H' for Hydrogen.")

        np_potential = normalized_plasma_potential(debris_radius, debye_length_m, mass_ratio)
        print(f'Calculated normalized plasma potential: {np_potential}')

    vd_norm = vd / vi_km_s  # Normalised debris velocity

    return [np_potential, vd_norm, G] , TU, vd_norm
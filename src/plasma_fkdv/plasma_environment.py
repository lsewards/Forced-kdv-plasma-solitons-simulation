import numpy as np
from scipy.optimize import brentq
from .constants import __me__, __mh__, __mo__

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

def forcing_space_debris(np_potential, vd, G, x, time):
    """ Calculates the forcing term for the normalised forced KdV equation
        Parameters: X IS XI AND TIME IS TAU
        Returns: Forcing term as a numpy array
    """
    return np_potential * np.exp(-1 * (((x - vd * time ) / G) ** 2))

def get_forcing_space_debris_args(plasma_prop, debris_size, debye_length_m, vi_km_s, np_potential, vd, G):
    """ Returns a list of pre-defined parameters for the forcing term calculation."""
    # Unless directly specified, uses default values from spreadsheet or calculation
    if np_potential is None:
        # Determine the ion species and set the corresponding ion mass and mass ratio for the plasma region
        if plasma_prop["Ion Species"] == "O":
            mi = __mo__
            mass_ratio = __me__ / __mo__
        elif plasma_prop["Ion Species"] == "H":
            mi = __mh__
            mass_ratio = __me__ / __mh__
        else:
            raise ValueError("Ion species not recognised. Please use 'O' for Oxygen or 'H' for Hydrogen.")

        np_potential = normalised_plasma_potential(debris_size, debye_length_m, mass_ratio)

    if vd is None:
        vd = plasma_prop["Debris Velocity Circular Orbit"]  # Debris velocity in km/s
    if G is None:
        G = debris_size / debye_length_m  # Normalised forcing width

    vd_norm = vd / vi_km_s  # Normalised debris velocity

    return [np_potential, vd_norm, G]

def initialise_ionosphere_region(df, region):
    """ takes a dataframe of ionospheric properties and pulls the associated properties for a given region.
    making the necessary calculations required to run simulation CURRENTLY NOT ABLE TO HANDLE GEO"""

    # Pulls property name and values into a list and numpy array respectively
    value = df[region].to_numpy()
    properties = df["Properties"].to_list()

    # Creates a dictionary of the properties and values for the given region
    plasma_properties = dict(zip(properties, value))
    return plasma_properties
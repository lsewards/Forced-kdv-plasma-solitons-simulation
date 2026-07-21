import numpy as np
from scipy.optimize import brentq

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
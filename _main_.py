## Imports & variables
import numpy as np
import pandas as pd
from functools import partial
import fkdv_simulation as sim
import plasma_enviroment as env
import exact_solutions as exact

# Global constants
__kb__ = 1.380649e-23  #Boltzmann constant in J/K
__c__ = 299792458.0  # Speed of light in m/s
__me__ = 9.1093837139e-31  # Electron mass in kg
__mh__ = 1.67262192595e-27  # Hydrogen ion aka proton mass in kg
__mo__ = 15.994914619 * 1.66053906892e-27  # Oxygen ion mass in kg

region = "Low LEO"  # Ionosphere region to simulate

# physical parameters
alpha = 6 # Nonlinearity parameter ( 1.0051)
beta = 1 # Dispersion parameter (0.4925)

# forcing parameters
forcing_fn = exact.unforced_kdv_solution
forcing_args = None
debris_size = 0.5 # Forcing amplitude in V
vd = None # Debris velocity in km/s
np_potential = None  # Normalised plasma potential, if None, it will be automatically calculated.
G = None  # Normalised forcing width

# simulation intialisation
mode = "analytical"  # simulation mode: "rest" for starting from rest frame, "analytical" for starting from an exact solution
exact_wave_fn = exact.n_soltion_exact  # module containing analytical solutions for the KdV equation
stabilising_steps = 1  # number of steps to stabilise the solution at the start of the simulation

# Simulation parameters
N = 1024  # number of discretised points in the domain (power of 2 for FFT speed)
L = 50.0  # size of the domain in debye lengths
dtau = 1e-4  # TU
tau_inital = 0.0  # TU
tau_final =  1.0   # TU

# Plotting params
verbose = 10  # print every verbose number of steps
time_stamp_2d = 30.0  # time stamp for the 2D plot in TU
view = [23,-110, 0]  # 3D plot view angles
x_lim = None
y_lim = None
u_lim = None
display_debris_pos = False

def initialise_ionosphere_region(df, region):
    """ takes a dataframe of ionospheric properties and pulls the associated properties for a given region.
    making the necessary calculations required to run simulation CURRENTLY NOT ABLE TO HANDLE GEO"""

    # Pulls property name and values into a list and numpy array respectively
    value = df[region].to_numpy()
    properties = df["Properties"].to_list()

    # Creates a dictionary of the properties and values for the given region
    plasma_properties = dict(zip(properties, value))
    return plasma_properties

if __name__ == "__main__":

    # Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
    df_ionosphere = pd.read_excel(r'Ionosphere prop.xltx')
    plasma_prop = initialise_ionosphere_region(df_ionosphere, region)

    # Normalizes plasma properties
    debye_length_m = plasma_prop["Debye Length"] * 1e-2
    vi_km_s = plasma_prop["Ion Acoustic Velocity"]
    TU = debye_length_m / (vi_km_s * 1e3)

    # Determine the ion species and set the corresponding ion mass and mass ratio
    if plasma_prop["Ion Species"] == "O":
        mi = __mo__
        mass_ratio = __me__ / __mo__
    elif plasma_prop["Ion Species"] == "H":
        mi = __mh__
        mass_ratio = __me__ / __mh__
    else:
        raise ValueError("Ion species not recognised. Please use 'O' for Oxygen or 'H' for Hydrogen.")

    # Unless directly speified, uses default values from spreadsheet or calculation
    if forcing_fn is None:
        if np_potential is None:
            np_potential = env.normalised_plasma_potential(debris_size, debye_length_m, mass_ratio)
        if vd is None:
            vd = plasma_prop["Debris Velocity Circular Orbit"]  # Debris velocity in km/s
        if G is None:
            G = debris_size / debye_length_m  # Normalised forcing width

        vd_norm = vd / vi_km_s  # Normalised debris velocity

        forcing_fn = env.forcing_space_debris
        forcing_args = [np_potential, vd_norm, G]

    if forcing_args:
        loaded_forcing_fn= partial(forcing_fn, *forcing_args)
    else:
        loaded_forcing_fn = forcing_fn

    sol, time_array, x_array = sim.run_simulation(L, N, tau_inital, tau_final, dtau, alpha, beta,
                                            mode=mode,
                                            forcing_fn=loaded_forcing_fn,
                                            analytical_sol= exact_wave_fn if mode == "analytical" else None,
                                            stabilising_steps=stabilising_steps,
                                            verbose=verbose,
                                            solver=sim.ck_solver
    )

    sim.view_simulation_results(sol, time_array, x_array,
                            time_stamp=time_stamp_2d,
                            debris_pos = display_debris_pos,
                            view = view,
                            x_lim = x_lim,
                            t_lim = y_lim,
                            u_lim = u_lim
    )
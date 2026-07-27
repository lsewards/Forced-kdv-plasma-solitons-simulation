## Imports & variables
import numpy as np
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import fkdv_simulation as sim
import plasma_enviroment as env
import exact_solutions as exact

importlib.reload(sim)
importlib.reload(env)
importlib.reload(exact)

# Global constants
__kb__ = 1.380649e-23  #Boltzmann constant in J/K
__c__  = 299792458.0  # Speed of light in m/s

region = "Low LEO"  # Ionosphere region to simulate

# physical parameters
alpha        = 1 # Nonlinearity parameter ( 1.0051)
beta         = 1/6 # Dispersion parameter (0.4925)
# below only relivent when forcing function is env.forcing_space_debris
debris_size  = 0.5
vd           = None
np_potential = None
G            = None


# ----- Simulation parameters ----
L         = 50.0
N         = 1024
dt        = 1e-4
t_initial = -2.0
t_final   =  2.0

mode = "analytical"  # "rest" for starting from rest frame, "analytical" for starting from an exact solution

exact_wave_fn = exact.xiao_traveling_wave_exact  # module containing analytical solutions for the KdV equation
forcing_fn = exact.xiao_traveling_forcing_exact

wave_args = [beta, 4, 1, [-1, 2, -1]]  # arguments for the analytical solution function
forcing_args = [-2, beta, 4, 1, [-1, 2, -1]]

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

    if wave_args:
        loaded_wave_exact = partial(exact_wave_fn, *wave_args)
    else:
        loaded_wave_exact = exact_wave_fn


    if forcing_fn == env.forcing_space_debris:
        forcing_args = env.get_forcing_space_debris_args(plasma_prop, debris_size, debye_length_m, vi_km_s,
                                                         np_potential, vd, G)

    elif forcing_fn == exact.xiao_traveling_forcing_exact or forcing_fn == exact.xiao_special_forcing_exact:
        loaded_forcing_fn = exact.normalised_xiao_forcing_exact(-2 , forcing_fn, *forcing_args)

    else:
        loaded_forcing_fn = partial(forcing_fn, *forcing_args) if forcing_args else forcing_fn


    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode=mode,
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              stabilising_steps=10,
                                                              verbose=10,
                                                              solver=sim.ck_solver
                                                              )

    # Plotting params
    time_stamp_2d = 0  # time stamp for the 2D plot in TU
    view = [23, -110, 0]  # 3D plot view angles
    x_lim = [-5, 5]
    t_lim = [-1, 1]
    display_debris_pos = False

    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                debris_pos=display_debris_pos,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )


    if exact_wave_fn is not None:
        # Compare the numerical solution with the analytical solution at a specific time stamp
        real_sol = sim.get_real_solutions(loaded_wave_exact, sim_x_array, sim_time_array)
        view_sim_results = sim.view_simulation_results(real_sol, sim_time_array, sim_x_array,
                                time_stamp=time_stamp_2d,
                                debris_pos = display_debris_pos,
                                view = view,
                                x_lim = x_lim,
                                t_lim = t_lim
        )

        i = sim.find_value_index(sim_time_array, time_stamp_2d)

        plt.figure(figsize=(10, 6))
        plt.plot(sim_x_array, real_sol[i] - sim_sol[i], color='blue')
        plt.title(f"Comparison of Numerical and Analytical Solutions at {time_stamp_2d} TU")
        plt.xlabel("x axis")
        plt.ylabel("Ureal - Uanalytical")
        plt.grid()
        plt.show()
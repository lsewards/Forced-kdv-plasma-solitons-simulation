## Imports & variables
import numpy as np
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from src.plasma_fkdv import fkdv_simulation as sim
from src.plasma_fkdv import exact_solutions as exact
from src.plasma_fkdv import plasma_environment as env
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__name__).resolve().parent.parent

# ======= Simulation Configuration =======

region = "Low LEO"  # Ionosphere region to simulate

#-------  Physical parameters ------------
alpha        = 1.0051 # Nonlinearity ( 1.0051)
beta         = 0.4925 # Dispersion (0.4925)
debris_size  = 0.1
vd           = None
np_potential = -1.77
G            = None


# -------- Simulation parameters ---------
L         = 2000.0
N         = 1024
dt        = 1e-4
t_initial =  0
t_final   =  200

mode = "rest"  # "rest" for starting from rest frame, "analytical" for starting from an exact solution

exact_wave_fn = None # module containing analytical solutions for the KdV equation
forcing_fn = env.forcing_space_debris # module containing forcing functions for the KdV equation

wave_args = None
forcing_args = None

# -------- Plotting params -----------------
time_stamp_2d = 100  # time stamp for the 2D plot in TU
view = [23, -110, 0]  # 3D plot view angles
x_lim = None
t_lim = None
display_debris_pos = False

if __name__ == "__main__":

    # Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
    df_ionosphere = pd.read_excel(PROJECT_ROOT / 'data' / 'Ionosphere_prop.xltx')
    plasma_prop = env.initialise_ionosphere_region(df_ionosphere, region)

    # Normalizes plasma properties
    debye_length_m = plasma_prop["Debye Length"] * 1e-2
    vi_km_s = plasma_prop["Ion Acoustic Velocity"]
    TU = debye_length_m / (vi_km_s * 1e3)

    forcing_args = env.get_forcing_space_debris_args(plasma_prop, debris_size, debye_length_m, vi_km_s,
                                                     np_potential, vd, G)
    loaded_forcing_fn = partial(forcing_fn, *forcing_args)

    if exact_wave_fn is not None:
        if wave_args is not None:
            loaded_wave_exact = partial(exact_wave_fn, *wave_args)
        else:
            loaded_wave_exact = exact_wave_fn
    else:
        loaded_wave_exact = exact_wave_fn




    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode=mode,
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              stabilising_steps=10,
                                                              verbose=1000,
                                                              solver=sim.ck_solver
    )

    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                debris_pos=display_debris_pos,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )
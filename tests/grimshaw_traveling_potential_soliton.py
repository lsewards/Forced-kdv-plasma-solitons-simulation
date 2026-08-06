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
alpha        = -3/2
beta         = -1/3
vd           = 0
np_potential = 0.1
G            = 2


# -------- Simulation parameters ---------
L         = 200.0
N         = 1024
dt        = 1e-4
t_initial =  0
t_final   =  200

mode = "rest"  # "rest" for starting from rest frame, "analytical" for starting from an exact solution

exact_wave_fn = None # module containing analytical solutions for the KdV equation
forcing_fn = env.forcing_space_debris # module containing forcing functions for the KdV equation

wave_args = None
forcing_args = [np_potential, vd, G ]

# -------- Plotting params -----------------
time_stamp_2d = 9 # time stamp for the 2D plot in TU
view = [20, -90, 0]  # 3D plot view angles
x_lim = [-50,50]
t_lim = None
display_debris_pos = False

if __name__ == "__main__":

    if wave_args is None:
        loaded_wave_exact = exact_wave_fn
    else:
        loaded_wave_exact = partial(exact_wave_fn, *wave_args)
    if forcing_args is None:
        loaded_forcing_fn = forcing_fn
    else:
        loaded_forcing_fn = partial(forcing_fn, *forcing_args)


    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode=mode,
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              stabilising_steps=10,
                                                              verbose=1000,
                                                              maker_solver=sim.make_ck_solver
    )

    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                debris_pos=display_debris_pos,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )
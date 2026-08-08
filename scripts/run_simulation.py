## Imports & variables
import numpy as np
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

exact_wave_fn = None
wave_args = None

#-------  Physical parameters ------------
alpha        = -3/2 # Nonlinearity ( 1.0051)
beta         = -1/6 # Dispersion (0.4925)
debris_radius  = 0.05
vd           = None
np_potential = -1.77
G            = None

# -------- Simulation parameters ---------
L         = 1500.0
N         = 13000
dt        = 1e-4
t_initial = 0
t_final   = 150
dealias   = True  # Apply the 2/3 rule to de-alias the nonlinear term

# -------- Plotting params -----------------
time_stamp_2d = 149.9 # time stamp for the 2D plot in TU
view = [23, -110, 0]  # 3D plot view angles
x_lim = None
t_lim = None
display_debris_pos = True

if __name__ == "__main__":

    # Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
    df_ionosphere = pd.read_excel(PROJECT_ROOT / 'data' / 'Ionosphere_prop.xltx')
    plasma_prop = env.initialise_ionosphere_region(df_ionosphere, region)

    debris_args, TU, vd_norm = env.get_forcing_space_debris_args(plasma_prop, debris_radius, np_potential, vd, G)
    loaded_forcing_fn = partial(env.forcing_space_debris, *debris_args)

    if wave_args:
        loaded_wave_exact = partial(exact_wave_fn, *wave_args)
    else:
        loaded_wave_exact = exact_wave_fn

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode='rest',
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              stabilising_steps=5,
                                                              verbose=100,
                                                              solver_maker=sim.make_ck_solver,
                                                              debris_args=debris_args,
                                                              dealias=dealias,
    )
    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                debris_pos=display_debris_pos,
                                vd_norm = vd_norm,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim,
                                plot_name=f"_{region.replace(' ', '_')}_a_{debris_radius}_np_potential_{np_potential}.png"
    )


    print(f'Maximum amplitude of the wave: {np.max(sim_sol)}')
    print(f'Minimum amplitude of the wave: {np.min(sim_sol)}')

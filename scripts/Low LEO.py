## Imports & variables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plasma_fkdv.utils import find_project_root
from plasma_fkdv import fkdv_sim as sim
from plasma_fkdv import plasma_env as env
from functools import partial

# ======= Simulation Configuration =======
region = "Low LEO" # INSERT REGION NAME HERE (e.g., "Low LEO", "High LEO I", "High LEO II"...)

#-------  Physical parameters ------------
alpha         = 1.0051 # Nonlinearity (1.0051)
beta          = 0.4925 # Dispersion (0.4925)
debris_radius = 0.05
vd            = None
np_potential  = None
G             = None

# -------- Simulation parameters ---------
L         = 6000.0
N         = 35000
dt        = 2e-3
t_initial = 0
t_final   = 200

# -------- Plotting params -----------------
time_stamp_2d = 200 # time stamp for the 2D plot
x_lim         = [-2000,2000]
t_lim         = None

if __name__ == "__main__":

    # Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
    df_ionosphere = pd.read_excel(find_project_root() / 'data' / 'Ionosphere_prop.xltx')
    plasma_prop = env.initialise_ionosphere_region(df_ionosphere, region)


    forcing_args, TU, vd_ratio = env.get_forcing_space_debris_args(plasma_prop, debris_radius, np_potential, vd, G)
    loaded_forcing_fn = partial(env.forcing_space_debris, *forcing_args)

    soliton_type = env.predict_soliton_type( vd_ratio, alpha, beta, forcing_args[0], forcing_args[2])
    print(f"Predicted soliton type: {soliton_type} (vd_ratio = {vd_ratio:.3f})")

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode='rest',
                                                              forcing_fn=loaded_forcing_fn,
                                                              stabilising_steps=100,
                                                              verbose=1000,
                                                              solver_maker=sim.make_ck_solver,
                                                              debris_args=forcing_args
    )

    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                vd_norm = forcing_args[1],
                                x_lim=x_lim,
                                t_lim=t_lim,
                                plot_name=f"_{region.replace(' ', '_')}_a_{debris_radius}_np_potential_{np_potential}.png"
    )

    print(f'Maximum amplitude of the wave: {np.max(sim_sol)}')
    print(f'Minimum amplitude of the wave: {np.min(sim_sol)}')
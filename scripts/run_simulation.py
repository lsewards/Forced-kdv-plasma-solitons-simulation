## Imports & variables
from plasma_fkdv import fkdv_sim as sim
from plasma_fkdv import plasma_env as env
from plasma_fkdv import find_project_root
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ======= Simulation Configuration =======
region = "High LEO I"  # INSERT REGION NAME HERE (e.g., "Low LEO", "High LEO I", "High LEO II"...)

#-------  Physical parameters ------------
alpha         = 1.0051 # Nonlinearity ( 1.0051)
beta          = 0.4925 # Dispersion (0.4925)
debris_radius = 0.5
vd            = None
np_potential  = None
G             = None

# -------- Simulation parameters ---------
L         = 700
N         = 1024
dt        = 1e-4
t_initial = 0
t_final   = 10

# -------- Plotting params -----------------
time_stamp_2d = 5 # time stamp for the 2D plot in TU
view          = None # 3D plot view angles
x_lim         = None
t_lim         = None

if __name__ == "__main__":

    # Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
    df_ionosphere = pd.read_excel( find_project_root()/ 'data' / 'Ionosphere_prop.xltx')
    plasma_prop = env.initialise_ionosphere_region(df_ionosphere, region)

    debris_arg, TU, vd_ratio = env.get_forcing_space_debris_args(plasma_prop, debris_radius, np_potential, vd, G)
    loaded_forcing_fn = partial(env.forcing_space_debris, *debris_arg)

    soliton_type = env.predict_soliton_type(vd_ratio, alpha, beta, debris_arg.np_potential, debris_arg.G)
    print(f"Predicted soliton type: {soliton_type} (vd_ratio = {vd_ratio:.3f})")

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode='rest',
                                                              forcing_fn=loaded_forcing_fn,
                                                              stabilising_steps=10,
                                                              verbose=1000,
                                                              solver_maker=sim.make_ck_solver,
                                                              debris_args=debris_arg,
    )
    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                vd_norm = debris_arg.vd_norm,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim,
                                plot_name=f"{region.replace(' ', '_')}_N{len(sim_x_array)}_a_{debris_radius}"
                                          f"_np_potential_{debris_arg.np_potential}.png"
    )

    print(f'Maximum amplitude of the wave: {np.max(sim_sol)}')
    print(f'Minimum amplitude of the wave: {np.min(sim_sol)}')

    ani = sim.generate_animated_plot(sim_sol, sim_time_array, sim_x_array,
                                     vd_norm=debris_arg.vd_norm,
                                     plot_name=f"{region.replace(' ', '_')}_N{len(sim_x_array)}_a_{debris_radius}"
                                               f"_np_potential_{debris_arg.np_potential}.gif"
    )
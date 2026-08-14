## Imports & variables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plasma_fkdv import fkdv_sim as sim
from plasma_fkdv import plasma_env as env
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =====================================================================
# ======= Simulation Configuration =======
region = 'High LEO II' # INSERT REGION NAME HERE (e.g., "Low LEO", "High LEO I", "High LEO II"...)

exact_wave_fn = None
wave_args = None

# Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
df_ionosphere = pd.read_excel(PROJECT_ROOT / 'data' / 'Ionosphere_prop.xltx')
plasma_prop = env.initialise_ionosphere_region(df_ionosphere,  region)

#-------  Physical parameters ------------
alpha         = 1.0051 # Nonlinearity ( 1.0051)
beta          = 0.4925 # Dispersion (0.4925)
debris_radius = 2.5e-3
vd            = None
np_potential  = -0.96
G             = None

# -------- Simulation parameters ---------
L         = 200
N         = 60000
dt        = 2e-3
t_initial = 0
t_final   = 240

# -------- Plotting params -----------------
time_stamp_2d = 240 # time stamp for the 2D plot in TU
x_lim = None
t_lim = None

#======================================================================
if __name__ == "__main__":

    forcing_args, TU, vd_ratio = env.get_forcing_space_debris_args(plasma_prop, debris_radius, np_potential, vd, G)
    loaded_forcing_fn = partial(env.forcing_space_debris, *forcing_args)

    soliton_type = env.predict_soliton_type(vd_ratio, alpha, beta, forcing_args.np_potential, forcing_args.G)
    print(f"Predicted soliton type: {soliton_type} (vd_ratio = {vd_ratio:.3f})")


    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode='rest',
                                                              forcing_fn=loaded_forcing_fn,
                                                              stabilising_steps=10,
                                                              verbose=1000,
                                                              solver_maker=sim.make_ck_solver,
                                                              debris_args=forcing_args,
                                                              checkpoint_every=20000
    )
    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                vd_norm = forcing_args.vd_norm,
                                x_lim=x_lim,
                                t_lim=t_lim,
                                plot_name=f"{region.replace(' ', '_')}_N{len(sim_x_array)}_a_{debris_radius}"
                                          f"_np_potential_{forcing_args.np_potential:.2f}.png"
    )
    print(f'Maximum amplitude of the wave: {np.max(sim_sol)}')
    print(f'Minimum amplitude of the wave: {np.min(sim_sol)}')
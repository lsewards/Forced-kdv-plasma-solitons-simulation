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

#-------  Physical parameters ------------
alpha        = -3/2 # Nonlinearity ( 1.0051)
beta         = -1/6 # Dispersion (0.4925)
# below only relevant when forcing function is env.forcing_space_debris
debris_radius  = 0.05
vd           = None
np_potential = -1.77
G            = None


# -------- Simulation parameters ---------
L         = 2000.0
N         = 2000
dt        = 1e-4
t_initial = 0
t_final   = 200

exact_wave_fn = None
forcing_fn = env.forcing_space_debris

wave_args = None

# -------- Plotting params -----------------
time_stamp_2d = 199.9 # time stamp for the 2D plot in TU
view = [23, -110, 0]  # 3D plot view angles
x_lim = [-1500, 1500]
t_lim = None
display_debris_pos = True

if __name__ == "__main__":

    # Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
    df_ionosphere = pd.read_excel(PROJECT_ROOT / 'data' / 'Ionosphere_prop.xltx')
    plasma_prop = env.initialise_ionosphere_region(df_ionosphere, region)

    # Normalizes plasma properties
    debye_length_m = plasma_prop["Debye Length"] * 1e-2
    vi_km_s = plasma_prop["Ion Acoustic Velocity"]
    TU = debye_length_m / (vi_km_s * 1e3)

    forcing_args = env.get_forcing_space_debris_args(plasma_prop, debris_radius, debye_length_m, vi_km_s, np_potential,
                                                     vd, G)
    loaded_forcing_fn = partial(forcing_fn, *forcing_args)


    if wave_args:
        loaded_wave_exact = partial(exact_wave_fn, *wave_args)
    else:
        loaded_wave_exact = exact_wave_fn

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode='rest',
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              stabilising_steps=10,
                                                              verbose=1000,
                                                              solver=sim.ck_solver
                                                              )

    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                debris_pos=display_debris_pos,
                                vd_norm = plasma_prop['Debris Velocity Circular Orbit'] / vi_km_s,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )

    print(f'Maximum amplitude of the wave: {np.max(sim_sol)}')
    print(f'Minimum amplitude of the wave: {np.min(sim_sol)}')


    if exact_wave_fn is not None:
        # Compare the numerical solution with the analytical solution at a specific time stamp
        real_sol = exact.get_real_solutions(loaded_wave_exact, sim_x_array, sim_time_array)
        view_sim_results = sim.view_simulation_results(real_sol, sim_time_array, sim_x_array,
                                time_stamp=time_stamp_2d,
                                debris_pos = display_debris_pos,
                                view = view,
                                x_lim = x_lim,
                                t_lim = t_lim
        )

        i = sim.find_value_index(sim_time_array, time_stamp_2d)

        plt.figure(figsize=(10, 6))
        plt.plot(sim_x_array, (real_sol[i] - sim_sol[i]), color='blue')
        plt.title(f"Comparison of Numerical and Analytical Solutions at {time_stamp_2d} TU")
        plt.xlabel("x axis")
        plt.ylabel("Ureal - Uanalytical")
        plt.grid()
        plt.show()

        print(f"At time {time_stamp_2d} t")
        print(f"The Absolute Error : {np.max(np.abs(real_sol[i] - sim_sol[i]))}")
        print(f"The Relative Error : {np.max(np.abs(real_sol[i] - sim_sol[i])) / np.max(np.abs(real_sol[i]))}")

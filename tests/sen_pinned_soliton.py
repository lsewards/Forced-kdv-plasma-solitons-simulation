## Imports & variables
import numpy as np
import matplotlib.pyplot as plt
from src.plasma_fkdv import fkdv_simulation as sim
from src.plasma_fkdv import exact_solutions as exact
from functools import partial

#-------  Physical parameters ------------
alpha        = -3/2 # Nonlinearity ( 1.0051)
beta         = -1/6 # Dispersion (0.4925)

# -------- Simulation parameters ---------
L         = 60.0
N         = 1024
dt        = 1e-4
t_initial = 0
t_final   = 10

exact_wave_fn = exact.sen_wave_exact
forcing_fn = exact.sen_forcing_exact

wave_args = [20, 1/2]
forcing_args = [10, 1/2]

# -------- Plotting params -----------------
time_stamp_2d = 1  # time stamp for the 2D plot in TU
view = [23, -110, 0]  # 3D plot view angles
x_lim = [10,20]
t_lim = None

if __name__ == "__main__":

    loaded_forcing_fn = partial(forcing_fn,*forcing_args)
    loaded_wave_exact = partial(exact_wave_fn, *wave_args)

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="analytical",
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              verbose=100,
                                                              solver=sim.ck_solver
    )




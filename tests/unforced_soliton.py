## Imports & variables
import numpy as np
import matplotlib.pyplot as plt
from src.plasma_fkdv import fkdv_simulation as sim
from src.plasma_fkdv import exact_solutions as exact

# ----- physical params ----------------
alpha     = 6 # Nonlinearity ( 1.0051)
beta      = 1 # Dispersion (0.4925)

# -------- Simulation parameters ---------
L         = 50.0
N         = 1024
dt        = 1e-4
t_initial = 0
t_final = 1

# -------- Plotting params -----------------
time_stamp_2d = 0.5  # time stamp for the 2D plot in TU
view = [23, -110, 0]  # 3D plot view angles
x_lim = None
t_lim = None

if __name__ == "__main__":

    exact_wave_fn = exact.n_soltion_exact  # module containing analytical solutions for the KdV equation
    forcing_fn = exact.unforced_kdv_solution  # module containing forcing functions for the KdV equation

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="analytical",
                                                              forcing_fn=forcing_fn,
                                                              analytical_sol=exact_wave_fn,
                                                              verbose=10,
                                                              solver_maker=sim.make_ck_solver
                                                              )
    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )
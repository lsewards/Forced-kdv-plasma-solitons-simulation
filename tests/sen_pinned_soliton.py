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
t_final   = 100

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
                                                              verbose=1000,
                                                              solver=sim.ck_solver
    )

    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )


    if exact_wave_fn is not None:
        # Compare the numerical solution with the analytical solution at a specific time stamp
        real_sol = exact.get_real_solutions(loaded_wave_exact, sim_x_array, sim_time_array)
        view_sim_results = sim.view_simulation_results(real_sol, sim_time_array, sim_x_array,
                                time_stamp=time_stamp_2d,
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
## Imports & variables
from plasma_fkdv import run_simulation, view_simulation_results, make_ck_solver, find_value_index
from plasma_fkdv import exact_solutions as exact
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

#-------  Physical parameters ------------
alpha        = 1 # Nonlinearity ( 1.0051)
beta         = 1/6 # Dispersion (0.4925)
alpha_zg     = -2 # Forcing term coefficient
kappa = 1
b = [-1, 2, -1]# Forcing term parameters
delta = 4

# -------- Simulation parameters ---------
L         = 50.0
N         = 1024
dt        = 1e-4
t_initial = -3
t_final   =  3

# -------- Plotting params -----------------
time_stamp_2d = 2 # time stamp for the 2D plot in TU
view = None  # 3D plot view angles
x_lim = [-5, 5]
t_lim = [-2, 2]

if __name__ == "__main__":

    wave_args = [beta, delta, kappa, b]
    forcing_args = [alpha_zg, beta, delta, kappa, b]

    loaded_forcing_fn = exact.normalised_xiao_forcing_exact(alpha_zg, exact.xiao_traveling_forcing_exact,*forcing_args)
    loaded_wave_exact = partial(exact.xiao_traveling_wave_exact, *wave_args)

    sim_sol, sim_time_array, sim_x_array = run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="analytical",
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              verbose=10,
                                                              solver_maker=make_ck_solver
    )
    view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )

    # Compare the numerical solution with the analytical solution at a specific time stamp
    real_sol = exact.get_real_solutions(loaded_wave_exact, sim_x_array, sim_time_array)
    view_sim_results = view_simulation_results(real_sol, sim_time_array, sim_x_array,
                            time_stamp=time_stamp_2d,
                            view = view,
                            x_lim = x_lim,
                            t_lim = t_lim
    )

    i = find_value_index(sim_time_array, time_stamp_2d)

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
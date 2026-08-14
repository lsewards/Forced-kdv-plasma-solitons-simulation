## Imports & variables
from plasma_fkdv import exact_solutions as  exact
from plasma_fkdv import fkdv_sim as sim
from plasma_fkdv.utils import find_value_index, find_project_root
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

#-------  Physical parameters ------------
alpha = 1  # Nonlinearity ( 1.0051)
beta = 1 / 2  # Dispersion (0.4925)
A = 20
G = 1/2

# -------- Simulation parameters ---------
L = 60.0
N = 2024
dt = 1e-4
t_initial = 0
t_final = 4

# -------- Plotting params -----------------
time_stamp_2d = 1  # time stamp for the 2D plot in TU
view = None # 3D plot view angles
x_lim = [13, 19]
t_lim = None

if __name__ == "__main__":

    wave_args, forcing_args = [A, G], [A, G]

    loaded_forcing_fn = partial(exact.sen_forcing_exact, *forcing_args)
    loaded_wave_exact = partial(exact.sen_wave_exact, *wave_args)

    # Run the simulation and compare the numerical solution with the analytical solution at a specific time stamp
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="analytical",
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol=loaded_wave_exact,
                                                              verbose=100,
                                                              solver_maker=sim.make_ck_solver
    )
    sim_sol, _, sim_x_array = sim.slice_array(sim_sol, sim_time_array, sim_x_array, x_lim=x_lim, t_lim=t_lim)

    i = find_value_index(sim_time_array, time_stamp_2d)
    real_sol = loaded_wave_exact(sim_x_array, sim_time_array[i])

    # Plot the difference between the numerical and analytical solutions at the specified time stamp
    plt.figure(figsize=(10, 6))
    plt.plot(sim_x_array, (real_sol - sim_sol[i]), color='blue')
    plt.title(f"Comparison of Numerical and Analytical Solutions at {time_stamp_2d} TU")
    plt.xlabel("x axis")
    plt.ylabel("Ureal - Uanalytical")
    plt.grid()
    plt.show()

    print(f"At time {time_stamp_2d} t")
    print(f"The Absolute Error : {np.max(np.abs(real_sol - sim_sol[i]))}")
    print(f"The Relative Error : {np.max(np.abs(real_sol - sim_sol[i])) / np.max(np.abs(real_sol))}")

    # Plot the evolution of the solution over 15 seconds on the same graph biased in y
    times = [0, 0.1, 0.2, 0.3]
    dashed = ['-', '--', '-.', ':']
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    u_lines = []

    for time in times:
        i = find_value_index(sim_time_array, time + 2.34)
        u_line = ax2.plot(sim_x_array, sim_sol[i], label=f't={time}', color='orange', linestyle=dashed[times.index(time)])
        u_lines.append(u_line)

        u = loaded_wave_exact(sim_x_array, sim_time_array[i])
        #ax2.plot(sim_x_array, u, label=f'Exact solution', color='orange')

        f = loaded_forcing_fn(sim_x_array, sim_time_array[i])
        f_line = ax1.plot(sim_x_array, f, color='blue')

    ax2.set_ylabel("U", color='orange')
    ax2.set_ylim([0, 25])
    ax2.tick_params(axis='y', labelcolor='orange')

    ax1.set_ylabel("Forcing term", color='blue')
    ax1.set_ylim([0, 10])
    ax1.tick_params(axis='y', labelcolor='blue')

    ax1.set_xlabel("x")
    ax1.grid()
    ax2.legend(handles=[f_line[0], u_lines[0][0], u_lines[1][0], u_lines[2][0], u_lines[3][0]],
               labels=['Moving Force', 't=0', 't=0.1', 't=0.2', 't=0.3'], loc='upper right')
    plt.show()

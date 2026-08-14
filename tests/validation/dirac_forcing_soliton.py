## Imports & variables
from plasma_fkdv import run_simulation, view_simulation_results, make_ck_solver, find_value_index, slice_array
from plasma_fkdv import exact_solutions as  exact
import matplotlib.pyplot as plt
from functools import partial

#-------  Physical parameters ------------
alpha        = -3/2 # Nonlinearity ( 1.0051)
beta         = -1/6 # Dispersion (0.4925)
dirac_amplitude = 1

# -------- Simulation parameters ---------
L         = 250.0
N         = 2000
dt        = 1e-4
t_initial = -1
t_final   =  16

# -------- Plotting params -----------------
time_stamp_2d = 2
view = None
x_lim = None
t_lim = None
dirac_plot = False

if __name__ == "__main__":

    loaded_forcing_fn = partial(exact.dirac_forcing_exact,[dirac_amplitude])


    sim_sol, sim_time_array, sim_x_array = run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="rest",
                                                              forcing_fn=loaded_forcing_fn,
                                                              verbose=1000,
                                                              solver_maker=make_ck_solver
    )
    view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp=time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )
    sim_sol, sim_time_array, sim_x_array = slice_array(sim_sol, sim_time_array, sim_x_array,
                                                           x_lim=x_lim,
                                                           t_lim=t_lim
    )

    plt.figure(figsize=(10, 6))
    for offset in range(1, 16):
        i = find_value_index(sim_time_array, offset )
        print(sim_time_array[i])
        plt.plot(sim_x_array, sim_sol[i] + 1 * offset)
    plt.title(f"U")
    plt.xlabel("x axis")
    plt.ylabel("U")
    plt.grid()
    plt.show()

    if dirac_plot:
        dirac = loaded_forcing_fn(sim_x_array, 0)

        plt.figure(figsize=(10, 6))
        plt.plot(sim_x_array, dirac, color='blue')
        plt.title(f"U")
        plt.xlabel("x axis")
        plt.ylabel("U")
        plt.grid()
        plt.show()

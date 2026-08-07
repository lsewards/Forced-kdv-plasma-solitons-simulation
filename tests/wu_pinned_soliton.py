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
L         = 70.0
N         = 2000
dt        = 1e-4
t_initial = 0
t_final   = 100

exact_wave_fn = exact.wu_wave_exact
forcing_fn = exact.wu_forcing_exact

a  = 0.5
b1 = -0.1250
k  = 0.6124
wave_args = [ a, k]
forcing_args = [ b1, k]

# -------- Plotting params -----------------
time_stamp_2d = .5  # time stamp for the 2D plot in TU
view = [40, -90, 0]  # 3D plot view angles
x_lim = [-40,40]
t_lim = None

if __name__ == "__main__":

    loaded_forcing_fn = partial(forcing_fn,*forcing_args)
    loaded_wave_exact = partial(exact_wave_fn, *wave_args)


    # analytical simulation

    anal_sol,  sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="analytical",
                                                              forcing_fn=loaded_forcing_fn,
                                                              analytical_sol= loaded_wave_exact,
                                                              verbose=1000,
                                                              solver_maker=sim.make_ck_solver
    )
    
    sim.view_simulation_results(anal_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )

    i = sim.find_value_index(sim_time_array, time_stamp_2d)

    u = loaded_wave_exact(sim_x_array, sim_time_array[i])
    plt.figure(figsize=(10, 6))
    plt.plot(sim_x_array, (u - anal_sol[i]), color='blue')
    plt.title(f"Comparison of Numerical and Analytical Solutions at {time_stamp_2d} TU")
    plt.xlabel("x axis")
    plt.ylabel("Urest - Uanalytical")
    plt.grid()
    plt.show()

    print(f"At time {time_stamp_2d} t")
    print(f"The Absolute Error : {np.max(np.abs(u - anal_sol[i]))}")
    print(f"The Relative Error : {np.max(np.abs(u - anal_sol[i])) / np.max(np.abs(anal_sol[i]))}")


    #Rest simulationm
    num_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="rest",
                                                              forcing_fn=loaded_forcing_fn,
                                                              stabilising_steps= 10,
                                                              verbose=100,
                                                              solver_maker=sim.make_ck_solver
    )
    sim.view_simulation_results(num_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )

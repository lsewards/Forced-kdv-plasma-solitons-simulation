## Imports & variables
import matplotlib.pyplot as plt
from src.plasma_fkdv import fkdv_simulation as sim
from src.plasma_fkdv.fkdv_simulation import find_value_index
from src.plasma_fkdv import exact_solutions as  exact
from functools import partial

exact_wave_fn = None
forcing_fn = exact.dirac_forcing_exact

wave_args = None
forcing_args = [1]

#-------  Physical parameters ------------
alpha        = -3/2 # Nonlinearity ( 1.0051)
beta         = -1/6 # Dispersion (0.4925)

# -------- Simulation parameters ---------
L         = 250.0
N         = 2000
dt        = 1e-4
t_initial = -1
t_final   =  16

# -------- Plotting params -----------------
x_lim = None
t_lim = None

if __name__ == "__main__":

    if wave_args is None:
        loaded_wave_exact = exact_wave_fn
    else:
        loaded_wave_exact = partial(exact_wave_fn, *wave_args)
    if forcing_args is None:
        loaded_forcing_fn = forcing_fn
    else:
        loaded_forcing_fn = partial(forcing_fn,*forcing_args)

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="rest",
                                                              forcing_fn=loaded_forcing_fn,
                                                              verbose=1000,
                                                              solver_maker=sim.make_ck_solver
    )
    sim_sol, sim_time_array, sim_x_array = sim.slice_array(sim_sol, sim_time_array, sim_x_array,
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

    dirac_plot = False
    if dirac_plot:
        dirac = loaded_forcing_fn(sim_x_array, 0)

        plt.figure(figsize=(10, 6))
        plt.plot(sim_x_array, dirac, color='blue')
        plt.title(f"U")
        plt.xlabel("x axis")
        plt.ylabel("U")
        plt.grid()
        plt.show()

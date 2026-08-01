## Imports & variables
import numpy as np
import matplotlib.pyplot as plt
from src.plasma_fkdv import fkdv_simulation as sim
import scipy.signal as signal
from functools import partial

from src.plasma_fkdv.fkdv_simulation import find_value_index


def dirac_forcing_exact(x, t, amplitude=1, center = 250):
    dx = x[1] - x[0]
    i = np.argmin(np.abs(x - center))
    val = np.zeros_like(x)
    val[i] = amplitude / dx
    return val

def scipy_dirac_forcing_exact(x, t, amplitude=1, center = 0):
    return signal.unit_impulse(len(x), idx=np.argmin(np.abs(x - center))) * amplitude

exact_wave_fn =None
forcing_fn = scipy_dirac_forcing_exact

wave_args = None
forcing_args = None

#-------  Physical parameters ------------
alpha        = -3/2 # Nonlinearity ( 1.0051)
beta         = -1/6 # Dispersion (0.4925)

# -------- Simulation parameters ---------
L         = 150.0
N         = 1024
dt        = 5e-5
t_initial = -1
t_final   = 16

# -------- Plotting params -----------------
view = [23, -110, 0]  # 3D plot view angles
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
                                                              solver=sim.ck_solver
    )
    sim_sol, sim_x_array, sim_time_array = sim.slice_array(sim_x_array, sim_sol, sim_time_array,
                                                           x_lim=x_lim,
                                                           t_lim=t_lim
    )

    plt.figure(figsize=(10, 6))
    for time in range(15):
        i = find_value_index(sim_time_array, time)
        plt.plot(sim_x_array, sim_sol[i] + 1 * time)
    plt.title(f"U")
    plt.xlabel("x axis")
    plt.ylabel("U")
    plt.grid()
    plt.show()


    dirac = scipy_dirac_forcing_exact(sim_x_array, 0)

    plt.figure(figsize=(10, 6))
    plt.plot(sim_x_array, dirac, color='blue')
    plt.title(f"U")
    plt.xlabel("x axis")
    plt.ylabel("U")
    plt.grid()
    plt.show()
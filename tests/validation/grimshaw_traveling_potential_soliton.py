## Imports & variables
from plasma_fkdv import run_simulation, view_simulation_results, make_ck_solver, find_value_index, slice_array
from plasma_fkdv import plasma_env as env
from functools import partial

#-------  Physical parameters ------------
alpha        = -3/2
beta         = -1/3
vd           = 0
np_potential = 0.1
G            = 2

# -------- Simulation parameters ---------
L         = 200.0
N         = 1024
dt        = 1e-4
t_initial =  0
t_final   =  200

# -------- Plotting params -----------------
time_stamp_2d = 9 # time stamp for the 2D plot in TU
view = None # 3D plot view angles
x_lim = [-50,50]
t_lim = None

if __name__ == "__main__":

    forcing_args = [np_potential, vd, G]
    loaded_forcing_fn = partial(env.forcing_space_debris, *forcing_args)

    sim_sol, sim_time_array, sim_x_array = run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="rest",
                                                              forcing_fn=loaded_forcing_fn,
                                                              stabilising_steps=10,
                                                              verbose=1000,
                                                              solver_maker=make_ck_solver
    )
    view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim
    )
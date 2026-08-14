## Imports & variables
from plasma_fkdv import run_simulation, view_simulation_results, make_ck_solver
from plasma_fkdv import exact_solutions as exact

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
view = None # 3D plot view angles
x_lim = None
t_lim = None

if __name__ == "__main__":

    sim_sol, sim_time_array, sim_x_array = run_simulation(L, N, t_initial, t_final, dt, alpha, beta,
                                                              mode="analytical",
                                                              forcing_fn=exact.unforced_kdv_solution ,
                                                              analytical_sol=exact.n_soltion_exact,
                                                              verbose=10,
                                                              solver_maker=make_ck_solver
    )
    view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp= time_stamp_2d,
                                view=view,
                                x_lim=x_lim,
                                t_lim=t_lim,
                                plot_name= "unforced_soliton_results.png"
    )
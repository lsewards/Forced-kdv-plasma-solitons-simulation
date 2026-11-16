from plasma_fkdv import region, fkdv_sim

# This script condenses down the parameters used for the simulations in the /verifacation folder, and runs the simulation for a given region with different values of the parameter 'a'. The results are then visualized and animated.
region_name = "low_leo"
a = [0.01, 0.05, 0.1, 0.5]

for a_val in a:
    sim_sol, sim_time_array, sim_x_array = region.run_region(region_name, a=a_val, view_results=True, animate_results=True)




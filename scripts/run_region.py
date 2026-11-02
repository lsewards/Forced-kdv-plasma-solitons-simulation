from plasma_fkdv import region, fkdv_sim


a = [0.01, 0.05, 0.1, 0.5]

for a_val in a:
    sim_sol, sim_time_array, sim_x_array = region.run_region("low_leo", a=a_val, view_results=True)




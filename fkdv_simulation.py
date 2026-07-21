
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt

def ck_solver(prev_sol, curr_sol, f, k, alpha, beta, dt, s = 1.0):
    """Use the Chan-Kerkhoven scheme to solve the forced KdV equation on a periodic domain."""

    k_fs = k[:len(k) // 2 + 1] # slicng k down 1024 (real) = 1024 // 2 + 1 (fourier space )

    # pre-calculating coefficients
    _ck_beta_fs = dt * beta * 1j * (k_fs ** 3) * (s ** 3)
    _ck_z_fs = dt * 1j * k_fs * s

    # Fourier transform of the last solution, current solution, and forcing term
    fft_prev = rfft(prev_sol)
    fft_sqrd_curr = rfft(curr_sol ** 2)
    fft_f = rfft(f)

    return irfft( ( 1/(1 - _ck_beta_fs) ) * ( (1 + _ck_beta_fs)*fft_prev - alpha*_ck_z_fs*fft_sqrd_curr + _ck_z_fs*fft_f ) )

def fw_solver(prev_sol, curr_sol, f, k, alpha, beta, dt, s = 1.0):
    """Use the Fornberg & Whitham scheme to solve the forced KdV equation on a periodic domain."""

    k_fs = k[:len(k) // 2 + 1]

    fft_f = rfft(f)
    fft_curr = rfft(curr_sol)

    ifft_f_ik = irfft(fft_f * 1j * k_fs)
    ifft_curr_ik = irfft(fft_curr * 1j * k_fs)
    ifft_curr_ik_3 = irfft(fft_curr * (1j ** 3) * (k_fs ** 3))

    return (prev_sol
            - 2 * dt * alpha * curr_sol * s * ifft_curr_ik
            - 2 * dt * beta * (s**3) * ifft_curr_ik_3
            + dt * s * ifft_f_ik)

## Running the Simulation
def run_simulation(L, N, t_inital, t_final, dt, alpha, beta, mode="rest", analytical_sol=None, forcing_fn=None, stabilising_steps=10, verbose=10, solver = ck_solver):
    """Runs the simulation for the normalised forced KdV equation using the Crank-Nicolson scheme. The simulation can be run in two modes: 'rest' for starting from rest frame, and 'analyitcal' starting from some exact solution for the kdv equation. The function returns the solution array, time array, and mesh space."""

    def _get_next_solution(previous, current, current_time, dt):
        """Helper function to compute the next solution using the Crank-Nicolson scheme."""
        f = forcing_fn(x_array, current_time)
        next = solver(previous, current, f, k, alpha, beta, dt)

        if not np.all(np.isfinite(next)): # prevent sim from coninuting after reachin an overflow error
          raise FloatingPointError(
              f"Non-finite solution at time ={current_time:.6g}; "
              f"max |u|={np.max(np.abs(current)):.3e}")

        return next

    def _stabilization(prev_sol, curr_sol, sim_time, dt):
        """Helper function to stabilise the solution for the first time step to avoid numerical instability."""
        initial_sol = prev_sol.copy()
        for _ in range(stabilising_steps):
            prev_sol, curr_sol = curr_sol, _get_next_solution(prev_sol, curr_sol, sim_time, dt / stabilising_steps)
            sim_time += dt / stabilising_steps
        #print(f"Stabilisation complete. Time: {sim_time:.2f} TU")
        return initial_sol, curr_sol, sim_time  # The two-level step needs states separated by exactly ``dt``.

    x_array = np.linspace(-L, L, N, endpoint=False) # Defining mesh space & wave numbers
    k = 2 * np.pi * np.fft.fftfreq(N, d=(x_array[1] - x_array[0]))  # Wave numbers for FFT , d = dx

    if mode == "rest":
        num_allocated_steps = int((abs(t_inital - t_final) // dt + 1 + stabilising_steps) // verbose)
    elif mode == "analytical":
        if analytical_sol is None:
            raise ValueError("Analytical solution function must be provided when mode is 'analytical'.")
        num_allocated_steps = int((abs(t_inital - t_final) // dt + 1 ) // verbose)
    else:
        raise ValueError("Invalid mode. Please choose 'rest' or 'analytical'.")

    time_array = np.zeros(num_allocated_steps)  # Pre-allocate time array for efficiency
    sol = np.zeros((num_allocated_steps, N))  # Pre-allocate solution array for efficiency


    step = 1  # Start from the second time step since the first two are already set
    recorded_steps = 0  # Counter for the number of recorded steps
    sim_time = t_inital  # start
    while sim_time < t_final:
        if sim_time == t_inital: # Initalising simulation
            if mode == "rest" :
                prev_sol, curr_sol, sim_time = _stabilization(np.zeros(N), np.zeros(N), sim_time, dt)  # Stabilise the solution for the first time step
            else:
                prev_sol, curr_sol = analytical_sol(x_array, sim_time - dt), analytical_sol(x_array, sim_time)  # Initial condition is set to the exact solution of the KdV equation

        next_sol = _get_next_solution(prev_sol, curr_sol, sim_time, dt)

        if step % verbose == 0:  # only prints and records every verbose number of steps
            sol[recorded_steps] = next_sol
            time_array[recorded_steps] = sim_time
            recorded_steps += 1  # Increment the recorded steps counter
            print(f"Time: {sim_time:.2f} TU")
            #print(f"Max value of solution: {np.max(sol[recorded_steps - 1]):.4f}, Min value of solution: {np.min(sol[recorded_steps - 1]):.4f}")

        prev_sol, curr_sol = curr_sol, next_sol
        step += 1
        sim_time += dt

    print("Simulation complete.")
    return sol[:recorded_steps], time_array[:recorded_steps], x_array

# real solution
def get_real_solutions(exact_wave_fn, x_array, time_array):
    real_sol = np.zeros((len(time_array), len(x_array)))
    for i in range(len(time_array)):
        real_sol[i] = exact_wave_fn(x_array, time_array[i])

    return real_sol

def find_time_index(time_array, time_tu):
    """Return the index of the recorded time closest to ``time_tu``."""
    if len(time_array) == 0:
        raise ValueError("Cannot select a time from an empty simulation result.")
    index = int(np.argmin(np.abs(time_array - time_tu)))
    print(f"Index of time step closest to {time_tu} TU: {index}")
    return index

def view_simulation_results(sol, time_array, x_array, time_stamp, debris_pos = False, view = [23,-110, 0], x_lim = None, t_lim = None, u_lim = None):
    """Visualizes the simulation results at a specific time step and as a 3D surface plot."""
    # Find the index of the time step closest to the specified time_tu
    print("Plotting simulation results...")

    plt.figure(figsize=(15, 7))

    # 2D plot of the solution at the specified time step
    index = find_time_index(time_array, time_stamp)

    ax = plt.subplot(121)
    ax.plot(x_array, sol[index], label=f"t = {time_array[index]:.2f} TU")
    #if debris_pos:
        #ax.axvline(vd_norm * time_array[index], color='red', linestyle='--', label="Debris position")
    ax.set_xlabel('x axis')
    ax.set_ylabel('U axis')
    ax.legend()
    ax.grid()

    # 3D surface plot of the solution over time
    Xi, Tau = np.meshgrid(x_array, time_array)  # Create meshgrid for plotting

    ax = plt.subplot(122, projection='3d')
    ax.plot_surface(Tau, Xi, sol, rstride=5, cstride=5, cmap='viridis')
    ax.set_xlabel('t axis')
    ax.set_ylabel('x axis')
    ax.set_zlabel('U axis')
    if t_lim is not None:
        ax.set_xlim(t_lim[0], t_lim[1])
    if x_lim is not None:
        ax.set_ylim(x_lim[0], x_lim[1])
    if u_lim is not None:
        ax.set_zlim(u_lim[0], u_lim[1])
    ax.view_init(elev=view[0], azim=view[1], roll=view[2])

    plt.show()
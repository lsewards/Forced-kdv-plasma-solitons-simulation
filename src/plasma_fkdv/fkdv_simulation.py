
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from tqdm import tqdm

def ck_solver(prev_sol, curr_sol, f, k, alpha, beta, dt):
    """Use the Chan-Kerkhoven scheme to solve the forced KdV equation on a periodic domain."""

    k_fs = k[:len(k) // 2 + 1] # slicing k down 1024 (real) = 1024 // 2 + 1 (fourier space )

    # pre-calculating coefficients
    _ck_beta_fs = dt * beta * 1j * (k_fs ** 3)
    _ck_z_fs = dt * 1j * k_fs

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
def run_simulation(L, N, t_initial, t_final, dt, alpha, beta, mode="rest", analytical_sol=None,
                   forcing_fn=None, stabilising_steps=10, verbose=10, solver = ck_solver):
    """Runs the simulation for the normalized forced KdV equation using the Chan-Kerkhoven scheme. The simulation can be
     run in two modes: 'rest' for starting from rest frame, and 'analytical' starting from some exact solution for the
     kdv equation. The function returns the solution array, time array, and mesh space."""

    def _get_next_solution(previous, current, current_time, time_step=dt):
        """Helper function to compute the next solution using the Chan-Kerkhoven scheme."""
        f = forcing_fn(x_array, current_time)
        next = solver(previous, current, f, k, alpha, beta, time_step)

        if not np.all(np.isfinite(next)): # prevent sim from continuing after reaching an overflow error
          raise FloatingPointError( f"Non-finite solution at time ={current_time:.6g}; " f"max |u|={np.max(np.abs(current)):.3e}")
        return next

    def _stabilization(previous, current, current_time , time_step=dt):
        """Helper function to stabilize the solution for the first time step to avoid numerical instability."""
        initial = previous.copy()
        for _ in range(stabilising_steps):
            previous, current = current, _get_next_solution( previous, current, current_time, time_step / stabilising_steps)
            current_time += time_step / stabilising_steps

        if abs((current_time - dt) - t_initial) > 1e-10:
            raise ValueError("Stabilisation failed to align with initial time.")
        else:
            return initial, current, current_time # The two-level step needs states separated by exactly ``dt``.

    if forcing_fn is None:
        raise ValueError("Forcing function must be provided.")

    x_array = np.linspace(-L, L, N, endpoint=False) # Defining mesh space & wave numbers
    k = 2 * np.pi * np.fft.fftfreq(N, d=(x_array[1] - x_array[0]))  # Wave numbers for FFT , d = dx

    # Calculating number steps simulation will do and pre-allocating arrays accordingly
    total_steps = int(round(abs(t_final - t_initial) / dt))
    num_allocated_steps = total_steps // verbose + 1
    time_array = np.zeros(num_allocated_steps)
    sol = np.zeros((num_allocated_steps, N))

    # --- initialize simulation state once, before the stepping loop
    sim_time = t_initial
    if mode == "rest":
        prev_sol, curr_sol, sim_time = _stabilization(np.zeros(N), np.zeros(N), sim_time)
        step = 1
    elif mode == "analytical":
        if analytical_sol is None:
            raise ValueError("Analytical solution function must be provided when mode is 'analytical'.")
        prev_sol, curr_sol = analytical_sol(x_array, sim_time - dt), analytical_sol(x_array, sim_time)
        step = 0
    else:
        raise ValueError("Invalid mode. Please choose 'rest' or 'analytical'.")

    sol[0] = curr_sol
    time_array[0] = sim_time
    recorded_steps = 1  # Counter for the number of recorded steps

    # ------- marching initial solution chosen forwards in time using the Chan-Kerkhoven scheme -------
    with tqdm(total = total_steps, initial = step, desc= "simulating") as pbar:
        while step < total_steps:
            next_sol = _get_next_solution( prev_sol, curr_sol, sim_time)
            sim_time += dt
            step += 1
            pbar.update(1)

            if step % verbose == 0:  # only records every verbose number of steps
                sol[recorded_steps] = next_sol
                time_array[recorded_steps] = sim_time
                recorded_steps += 1

            prev_sol, curr_sol = curr_sol, next_sol

    print("Simulation complete.")
    return sol[:recorded_steps], time_array[:recorded_steps], x_array

def find_value_index(array, value):
    """Return the index of the closest entry to value in an array."""
    if len(array) == 0:
        raise ValueError("Cannot select a value from an empty array.")
    if value < array[0] or value > array[-1]:
        raise ValueError(f"Value {value} is out of bounds for the array with min {array[0]} and max {array[-1]}.")
    return int(np.argmin(np.abs(array - value)))

def slice_array(u_arr, t_arr, x_arr, x_lim=None, t_lim=None):
    """Slices the array based on the provided limits for x, t, and u."""
    if (x_lim is not None and x_lim[0] > x_lim[1]) or (t_lim is not None and t_lim[0] > t_lim[1]):
        raise ValueError("Limits must be in the form [min, max].")

    if x_lim is not None:
        x_min_index = find_value_index(x_arr, x_lim[0])
        x_max_index = find_value_index(x_arr, x_lim[1])

        x_arr = x_arr[x_min_index:x_max_index + 1]
        u_arr = u_arr[:, x_min_index:x_max_index + 1]

    if t_lim is not None:
        t_min_index = find_value_index(t_arr, t_lim[0])
        t_max_index = find_value_index(t_arr, t_lim[1])

        t_arr = t_arr[t_min_index:t_max_index + 1]
        u_arr = u_arr[t_min_index:t_max_index + 1, :]

    return u_arr, t_arr, x_arr

def view_simulation_results(sol, time_array, x_array, time_stamp, debris_pos = False, vd_norm = 1.0,
                            view = [23,-110, 0], x_lim = None, t_lim = None):
    """Visualizes the simulation results at a specific time step and as a 3D surface plot."""

    if x_lim is not None or t_lim is not None:
        sol, time_array, x_array = slice_array(sol, time_array, x_array, x_lim=x_lim, t_lim=t_lim)

    t_index = find_value_index(time_array, time_stamp)

    print("Plotting simulation results...")

    plt.figure(figsize=(15, 7))

    ax = plt.subplot(121)
    ax.plot(x_array, sol[t_index], label=f"t = {time_array[t_index]:.2f} TU")
    if debris_pos:
        ax.axvline(vd_norm * time_array[t_index], color='red', linestyle='--', label="Debris position")
    ax.set_xlabel('x axis')
    ax.set_ylabel('U axis')
    ax.legend()
    ax.grid()

    # 3D surface plot of the solution over time
    Xi, Tau = np.meshgrid( x_array, time_array)  # Create meshgrid for plotting

    ax = plt.subplot(122, projection='3d')
    ax.plot_surface(Xi, Tau, sol, rstride=5, cstride=5, cmap='viridis')
    ax.set_xlabel('x axis')
    ax.set_ylabel('t axis')
    ax.set_zlabel('U axis')
    ax.view_init(elev=view[0], azim=view[1], roll=view[2])

    plt.show()
import inspect
import os
import subprocess
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Saved plots are committed automatically; set FKDV_AUTOCOMMIT_PLOTS=0 to turn this off.
AUTO_COMMIT_PLOTS = os.environ.get("FKDV_AUTOCOMMIT_PLOTS", "1") != "0"

def _two_thirds_mask(k):
    """Boolean mask over rfft wavenumbers implementing the 2/3 rule: wavenumbers above 2/3
    of the Nyquist wavenumber are zeroed so that aliasing errors from quadratic nonlinear
    terms fold into the discarded band instead of contaminating the resolved modes."""
    return k <= (2.0 / 3.0) * k[-1]

def _safe_filename_stem(caller_filename):
    """Stem of caller_filename with characters illegal in Windows filenames stripped, so that
    interactive callers (e.g. IPython's synthetic '<ipython-input-N-hash>' frame names) don't
    break file creation."""
    stem = Path(caller_filename).stem
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "")
    return stem or "interactive"

def _routed_dir(project_path, subdir_name, caller_filename):
    """Resolve project_path/tests/<subdir_name> or project_path/scripts/<subdir_name>
    depending on which tree the caller script lives in, mirroring the existing
    tests/plots and scripts/plots layout; falls back to project_path/<subdir_name>."""
    caller_file = Path(caller_filename).resolve()
    if "tests" in caller_file.parts:
        return project_path / "tests" / subdir_name
    elif "scripts" in caller_file.parts:
        return project_path / "scripts" / subdir_name
    else:
        return project_path / subdir_name

def find_value_index(array, value):
    """Return the index of the closest entry to value in an array."""
    if len(array) == 0:
        raise ValueError("Cannot select a value from an empty array.")
    if value < array[0] or value > array[-1]:
        print(f"Warning: Value {value} is outside range {array[0]} to {array[-1]}. Clipping to the nearest boundary.")
    return int(np.argmin(np.abs(array - value)))

def commit_plot(plot_path):
    """Stage and commit a single saved plot. Only that one path is staged, so unrelated
    work in progress is left untouched. Failures (no git, no repo, nothing to commit
    because the plot is byte-identical to the committed one) are reported and ignored:
    a simulation run should never die because of bookkeeping."""
    plot_path = Path(plot_path)
    try:
        rel = plot_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        print(f"Not committing {plot_path}: outside the project tree.")
        return

    def git(*args):
        return subprocess.run(("git", *args), cwd=PROJECT_ROOT,
                              capture_output=True, text=True)

    try:
        added = git("add", "--", rel)
        if added.returncode != 0:
            print(f"Could not stage {rel}: {added.stderr.strip()}")
            return

        # Empty diff against HEAD means the plot is unchanged, so there is nothing to commit.
        if git("diff", "--cached", "--quiet", "--", rel).returncode == 0:
            return

        committed = git("commit", "-m", f"Add plot {Path(rel).name}", "--only", "--", rel)
        if committed.returncode != 0:
            print(f"Could not commit {rel}: {(committed.stderr or committed.stdout).strip()}")
        else:
            print(f"Committed {rel}")
    except (OSError, subprocess.SubprocessError) as err:
        print(f"Could not commit {rel}: {err}")

def get_save_path(project_path, plot_name, caller_filename):
    plots_dir = _routed_dir(project_path, "plots", caller_filename)
    stem = plot_name[:-4] if plot_name.endswith(".png") else plot_name

    return plots_dir, stem
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

def make_ck_solver( N, dx, alpha, beta, dt, dealias=False):
    """Use the Chan-Kerkhoven scheme to solve the forced KdV equation on a periodic domain.
    If dealias is True, the 2/3 rule is applied to the nonlinear term to remove aliasing errors."""

    k = 2 * np.pi * np.fft.rfftfreq(N, d=dx)  # Wave numbers for FFT , d = dx
    mask = _two_thirds_mask(k) if dealias else None

    # pre-calculating coefficients
    b = dt * beta * 1j * (k ** 3)
    c0 = 1.0 / (1.0 - b)
    c1 = ( 1.0 + b ) * c0
    z = c0 * dt * 1j * k

    def solver(prev_sol, curr_sol, f):
        """Compute the next solution using the Chan-Kerkhoven scheme."""
        if mask is None:
            nonlinear_spec = rfft(curr_sol ** 2)
        else:
            # 2/3 rule: truncate before squaring so the aliases of the quadratic term land in
            # the discarded band, then truncate again to drop the product's own high-k content.
            nonlinear_spec = rfft(irfft(rfft(curr_sol) * mask) ** 2) * mask
        return irfft(c1 * rfft(prev_sol) - alpha * z * nonlinear_spec + z * rfft(f))
    return solver

def make_fw_solver( N, dx, alpha, beta, dt, dealias=False):
    """Use the Fornberg & Whitham scheme to solve the forced KdV equation on a periodic domain.
    If dealias is True, the 2/3 rule is applied to the nonlinear term to remove aliasing errors."""

    k = 2 * np.pi * np.fft.rfftfreq(N, d=dx)
    mask = _two_thirds_mask(k) if dealias else None

    def solver(prev_sol, curr_sol, f):

        curr_spec = rfft(curr_sol)
        nonlinear_spec = curr_spec * mask if mask is not None else curr_spec
        curr_sol_nonlinear = irfft(nonlinear_spec) if mask is not None else curr_sol

        ifft_f_ik = irfft(rfft(f) * 1j * k)
        ifft_curr_ik = irfft(nonlinear_spec * 1j * k)
        ifft_curr_ik_3 = irfft(curr_spec * (1j ** 3) * (k ** 3))

        return  (prev_sol
            - 2 * dt * alpha * curr_sol_nonlinear * ifft_curr_ik
            - 2 * dt * beta * ifft_curr_ik_3
            + dt  * ifft_f_ik)

    return solver

## Running the Simulation
def run_simulation(L, N, t_initial, t_final, dt, alpha, beta, mode="rest", analytical_sol=None,
                   forcing_fn=None, stabilising_steps=10, verbose=10, solver_maker = make_ck_solver,
                   debris_args = None, dealias=False, checkpoint_every=None, resume_from=None,
                   growth_warn_factor=10.0):
    """Runs the simulation for the normalized forced KdV equation using the Chan-Kerkhoven scheme. The simulation can be
     run in two modes: 'rest' for starting from rest frame, and 'analytical' starting from some exact solution for the
     kdv equation. The function returns the solution array, time array, and mesh space."""

    def _get_next_solution(solver, previous, current, current_time):
        """Helper function to compute the next solution using the Chan-Kerkhoven scheme."""
        f = forcing_fn(x_array, current_time)
        return solver(previous, current, f)

    def _stabilization(previous, current, current_time , time_step=dt):
        """Helper function to stabilize the solution when simulation is in 'rest' mode for the first time step to avoid
        numerical instability. The initial and current solutions need separated by exactly ``dt``. """
        initial = previous.copy()
        stab_solver = solver_maker(N, dx, alpha, beta, time_step / stabilising_steps, dealias=dealias)
        for _ in range(stabilising_steps):
            previous, current = current, _get_next_solution(stab_solver, previous, current, current_time)
            current_time += time_step / stabilising_steps
        return initial, current, current_time # The two-level step needs states separated by exactly ``dt``.

    x_array = np.linspace(-L, L, N, endpoint=False)  # Defining mesh space & wave numbers
    dx = x_array[1] - x_array[0]

    sim_solver = solver_maker(N, dx, alpha, beta, dt, dealias=dealias)

    if forcing_fn is None:
        raise ValueError("Forcing function must be provided.")
    if debris_args is not None and debris_args[-1] / dx < 8:
        print("Gaussian isnt well defined by chosen N & L")
    if debris_args is not None and debris_args[1] * t_final > L:
        print("L is too small, the forcing function will not be fully contained in the domain for the entire simulation.")

    checkpoint_path = None
    if checkpoint_every:
        caller_filename = inspect.stack()[1].filename
        checkpoint_dir = _routed_dir(PROJECT_ROOT, "checkpoints", caller_filename)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{_safe_filename_stem(caller_filename)}_L{L:g}_N{N}_dt{dt:g}.npz"

    def _save_checkpoint(previous, current, current_time, step_count):
        """Atomically write the two-level state needed to resume the march via resume_from=."""
        tmp_path = checkpoint_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, prev_sol=previous, curr_sol=current, sim_time=current_time, step=step_count,
                 L=L, N=N, dt=dt, alpha=alpha, beta=beta, dealias=dealias)
        os.replace(tmp_path, checkpoint_path)

    # Calculating number steps simulation will do and pre-allocating arrays accordingly
    total_steps = int(round(abs(t_final - t_initial) / dt))
    num_allocated_steps = total_steps // verbose + 1
    time_array, sol = np.zeros(num_allocated_steps), np.zeros((num_allocated_steps, N))

    # --- initialize simulation state once, before the stepping loop
    sim_time = t_initial
    if resume_from is not None:
        ckpt = np.load(resume_from)
        mismatches = [name for name, val in (("L", L), ("N", N), ("dt", dt), ("alpha", alpha), ("beta", beta))
                      if not np.isclose(float(ckpt[name]), float(val))]
        if bool(ckpt["dealias"]) != dealias:
            mismatches.append("dealias")
        if mismatches:
            raise ValueError(f"Checkpoint config mismatch for {mismatches}; cannot resume from {resume_from}.")
        prev_sol, curr_sol = ckpt["prev_sol"], ckpt["curr_sol"]
        sim_time, step = float(ckpt["sim_time"]), int(ckpt["step"])
    elif mode == "rest":
        prev_sol, curr_sol, sim_time = _stabilization(np.zeros(N), np.zeros(N), sim_time)
        step = 1
    elif mode == "analytical":
        if analytical_sol is None:
            raise ValueError("Analytical solution function must be provided when mode is 'analytical'.")
        prev_sol, curr_sol = analytical_sol(x_array, sim_time - dt), analytical_sol(x_array, sim_time)
        step = 0
    else:
        raise ValueError("Invalid mode. Please choose 'rest' or 'analytical'.")

    sol[0], time_array[0]  = curr_sol, sim_time
    recorded_steps = 1  # Counter for the number of recorded steps
    last_check_max = max(np.max(np.abs(curr_sol)), 1e-300)  # baseline for the amplitude growth warning

    # ------- marching initial solution chosen forwards in time using the Chan-Kerkhoven scheme -------
    with tqdm(total = total_steps, initial = step, desc= "Simulating") as pbar:
        while step < total_steps:
            next_sol = _get_next_solution( sim_solver, prev_sol, curr_sol, sim_time)
            sim_time += dt
            step += 1
            pbar.update(1)

            if step % verbose == 0:  # only records every verbose number of steps
                if not np.all(np.isfinite(next_sol)):  # prevent sim from continuing after reaching an overflow error
                    caller_filename = inspect.stack()[1].filename
                    crash_dir = _routed_dir(PROJECT_ROOT, "crash_dumps", caller_filename)
                    crash_dir.mkdir(parents=True, exist_ok=True)
                    crash_path = crash_dir / f"{_safe_filename_stem(caller_filename)}_L{L:g}_N{N}_dt{dt:g}_t{sim_time:.6g}.npz"
                    np.savez(crash_path, sol=sol[:recorded_steps], time_array=time_array[:recorded_steps],
                             x_array=x_array, L=L, N=N, dt=dt, alpha=alpha, beta=beta, dealias=dealias,
                             crash_time=sim_time)
                    tqdm.write(f"Non-finite solution at time={sim_time:.6g}; max |u|={np.max(np.abs(next_sol)):.3e}; "
                               f"stopping early and returning the partial trajectory (also saved to {crash_path}).")
                    break

                current_max = np.max(np.abs(next_sol))
                if current_max > growth_warn_factor * last_check_max:
                    tqdm.write(f"Warning: |u| grew {current_max / last_check_max:.1f}x over the last {verbose} "
                               f"steps (now {current_max:.3e}) at t={sim_time:.6g}; possible onset of instability.")
                last_check_max = current_max

                sol[recorded_steps] = next_sol
                time_array[recorded_steps] = sim_time
                recorded_steps += 1

            if checkpoint_path is not None and step % checkpoint_every == 0:
                _save_checkpoint(curr_sol, next_sol, sim_time, step)

            prev_sol, curr_sol = curr_sol, next_sol

    return sol[:recorded_steps], time_array[:recorded_steps], x_array

def view_simulation_results(sol, time_array, x_array, time_stamp, debris_pos = False, vd_norm = 1.0,
                            view = [23,-110, 0], x_lim = None, t_lim = None, plot_name = None):
    """Visualizes the simulation results at a specific time step and as a 3D surface plot."""

    N = len(x_array)

    if x_lim is not None or t_lim is not None:
        sol, time_array, x_array = slice_array(sol, time_array, x_array, x_lim=x_lim, t_lim=t_lim)

    t_index = find_value_index(time_array, time_stamp)

    print("Plotting simulation results...")
    plt.figure(figsize=(15, 7))

    ax = plt.subplot(121)
    ax.plot(x_array, sol[t_index], label=f"t = {time_array[t_index]:.2f} TU")
    if debris_pos:
        ax.axvline(time_array[0] + (vd_norm * time_array[t_index]), color='red', linestyle='--', label="Debris position")
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

    plt.tight_layout()

    if plot_name is not None:
        plots_dir, stem = get_save_path(PROJECT_ROOT, plot_name, inspect.stack()[1].filename)
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plots_dir / f"N{len(x_array)}{stem}.png"
        plt.savefig(plot_path, dpi='figure')
        print(f"Plot saved as {plot_path}")

        if AUTO_COMMIT_PLOTS:
            commit_plot(plot_path)

    plt.show()
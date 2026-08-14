from pathlib import Path
import numpy as np

def find_project_root(marker="pyproject.toml"):
    """Finds the project root using the marker file (default is pyproject.toml) and returns the path to the project root."""
    path = Path.cwd()
    for candidate in [path, *path.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate project root (no {marker} found above {path})")

def find_value_index(array, value):
    """Return the index of the closest entry to value in an array."""
    if len(array) == 0:
        raise ValueError("Cannot select a value from an empty array.")
    if value < array[0] or value > array[-1]:
        print(f"Warning: Value {value} is outside range {array[0]} to {array[-1]}. Clipping to the nearest boundary.")
    return int(np.argmin(np.abs(array - value)))

def get_save_path(project_path, plot_name, caller_filename):
    plots_dir = _routed_dir(project_path, "plots", caller_filename)
    stem = plot_name[:-4] if plot_name.endswith(".png") else plot_name
    return plots_dir, stem

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
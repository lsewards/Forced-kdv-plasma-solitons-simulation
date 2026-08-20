from plasma_fkdv import fkdv_sim as sim
from plasma_fkdv import plasma_env as env
from plasma_fkdv import find_project_root
from dataclasses import dataclass, fields
from functools import partial
import pandas as pd
import numpy as np
import tomllib

@dataclass
class RegionConfig:
    region: str                      # Excel column name, e.g. "High LEO I"
    a: float
    L: float
    N: int
    t_final: float
    sheet: str | None = None         # None → first sheet
    vd_mult: float | None = None     # multiples of the ion acoustic speed in the region
    vd: float | None = None           # speed of the debris in the region (km/s)
    np_potential: float | None = None
    G: float | None = None
    alpha: float = 1.0051       # dispersivity
    beta: float = 0.4925       # nonlinearity
    dt: float = 2e-3
    t_initial: float = 0.0
    stabilising_steps: int = 10
    time_stamp_2d: float | None = None   # None → t_final
    view: tuple[int, int, int] | None = None
    x_lim: list | None = None
    t_lim: list | None = None

def load_config(name, **overrides):
    with open(find_project_root() / 'config' / 'region.toml', 'rb') as f:
        config_data = tomllib.load(f)

    region_config_data = config_data.get(name)
    if region_config_data is None:
        raise ValueError(f"Region '{name}' not found in the configuration file.")

    merged = {**config_data.get('defaults', {}), **region_config_data, **overrides}

    valid_fields = {f.name for f in fields(RegionConfig)}
    unknown = set(merged) - valid_fields
    if unknown:
        raise ValueError(f"Unknown field(s) for region '{name}': {sorted(unknown)}")

    return RegionConfig(**merged)


def run_region(name, **overrides):

    config = load_config(name, **overrides)

    # Initializes the conditions for the region in question and sets the ion mass and mass ratio based on the ion species
    df_ionosphere = pd.read_excel(find_project_root() / 'data' / 'Ionosphere_prop.xltx', sheet_name=config.sheet)
    plasma_prop = env.initialise_ionosphere_region(df_ionosphere, config.region)

    if config.vd_mult is not None:
        config.vd = config.vd_mult * plasma_prop['Ion Acoustic Velocity']

    debris_arg, TU, vd_ratio = env.get_forcing_space_debris_args(plasma_prop, config.a, config.np_potential, config.vd,
                                                                 config.G)
    loaded_forcing_fn = partial(env.forcing_space_debris, *debris_arg)

    soliton_type = env.predict_soliton_type(vd_ratio, config.alpha, config.beta, debris_arg.np_potential, debris_arg.G)
    print(f"Predicted soliton type: {soliton_type} (vd_ratio = {vd_ratio:.3f})")

    # Run the simulation and view results
    sim_sol, sim_time_array, sim_x_array = sim.run_simulation(config.L, config.N, config.t_initial, config.t_final,
                                                              config.dt, config.alpha, config.beta,
                                                              mode='rest',
                                                              forcing_fn=loaded_forcing_fn,
                                                              stabilising_steps=10,
                                                              verbose=1000,
                                                              solver_maker=sim.make_ck_solver,
                                                              debris_args=debris_arg,
                                                              )
    sim.view_simulation_results(sim_sol, sim_time_array, sim_x_array,
                                time_stamp=config.time_stamp_2d,
                                vd_norm=debris_arg.vd_norm,
                                view=config.view,
                                x_lim=config.x_lim,
                                t_lim=config.t_lim,
                                plot_name=f"{config.region.replace(' ', '_')}_N{len(sim_x_array)}_a_{config.a}"
                                          f"_np_potential_{debris_arg.np_potential}.png"
                                )

    print(f'Maximum amplitude of the wave: {np.max(sim_sol)}')
    print(f'Minimum amplitude of the wave: {np.min(sim_sol)}')

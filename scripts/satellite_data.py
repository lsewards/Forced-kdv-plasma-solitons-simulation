import cdflib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_project_root(marker="pyproject.toml"):
    path = Path.cwd()
    for candidate in [path, *path.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate project root (no {marker} found above {path})")

def load_cdf_data(file_name):

    data_path = find_project_root() / "data" / f"{file_name}.cdf"

    cdf = cdflib.CDF(str(data_path))
    info = cdf.cdf_info()
    return cdf, info

plasma_cdf, plasma_info = load_cdf_data("SW_EXTD_EFIA_LP_HM_20131202T101113_20131202T140109_0102")


epoch = plasma_cdf.varget("Timestamp")
times = cdflib.cdfepoch.to_datetime(epoch)  # CDF_EPOCH -> numpy

density = plasma_cdf.varget("n")  # plasma density, per ccm
print(plasma_cdf.varattsget("n"))
altitude = plasma_cdf.varget("height")  # altitude, km
print(altitude)


print(np.max(altitude), np.min(altitude))

plt.figure(figsize=(12, 6))
plt.plot(times, density, label="Plasma Density")
plt.xlabel("Time")
plt.ylabel("Density (per ccm)")
plt.title("Plasma Density over Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


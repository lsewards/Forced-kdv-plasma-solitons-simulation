from plasma_fkdv import normalised_plasma_potential, forcing_space_debris
from plasma_fkdv.constants import  __me__, __mh__, __mo__
import matplotlib.pyplot as plt
import numpy as np

mass_ratio_o = __me__ / __mo__
mass_ratio_h = __me__ / __mh__

def validate_region(name, debye_length, mass_ratio, expected_phi_values):
    print(f"=Normalised plasma potential calculations for {name} region=:")
    for a, expected in zip(debris_radii, expected_phi_values):
        phi = normalised_plasma_potential(a, debye_length, mass_ratio)
        uncertainty = abs(phi - expected)
        print(f"Debris radius: {a:.1e} m, Normalised plasma potential: {phi:.3f}, uncertainty: {uncertainty:.3f}")
    print()


debye_length = 1  # Example value for debye radius

a_array = np.linspace(1e-3, 10, 16)  # Array of debris radii from 0.00001 to 10 in 16 steps

# # Plasma Potential Analysis
# ## Recreating figure 1 from NASA paper

phi_ox = np.array([normalised_plasma_potential(a, debye_length, mass_ratio_o) for a in a_array])
phi_h = np.array([normalised_plasma_potential(a, debye_length, mass_ratio_h) for a in a_array])

# PLOTTING
plt.figure(figsize=(8, 6))
plt.grid()
plt.plot(a_array / debye_length, phi_h, label="Hydrogen ion", color="blue")
plt.plot(a_array / debye_length, phi_ox, label="Oxygen ion", color="red")
plt.xlabel("debris radius / debye length")
plt.ylabel("Normalised plasma potential")
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(-2, 1, 0.1))
plt.xlim(0, 10)
plt.ylim(-1.9, -0.9)
plt.legend()
plt.show()

# ### Validating the normalised plasma potential calculation against NASA paper values (Table 6)
# values extracted from NASA paper Table 6

debris_diameters = [10e-2, 5e-2, 1e-2, 5e-3, 1e-3]  # Debris diameters in meters
debye_lengths = [.83e-2, 1.38e-2, 4.36e-2, 23504e-2]  # Debye lengths for Low LEO, Low LEO I, Low LEO II, and GEO respectively in meters
debris_radii = [d / 2 for d in debris_diameters]  # Debris radii in meters

lowleo_phi_values = [-1.77, -1.68, -1.40, -1.35, -1.33]
lowleoi_phi_values = [-1.20, -1.08, -0.94, -0.93, -0.92]
lowleoii_phi_values = [-1.02, -0.96, -0.92, -0.92, -0.92]

validate_region("Low LEO (Oxygen)", debye_lengths[0], mass_ratio_o, lowleo_phi_values)
validate_region("Low LEO I (Hydrogen)", debye_lengths[1], mass_ratio_h, lowleoi_phi_values)
validate_region("Low LEO II (Hydrogen)", debye_lengths[2], mass_ratio_h, lowleoii_phi_values)


# ### Calculating forcing term in Low LEO

a = 0.01  # Debris radius in m
L = 2 * np.pi  # size of the domain
vd = 0
N = 1024

tau = np.linspace(0, L, N)
xi = np.linspace(-3 * L, L, N)

e = (np.pi / L) * (xi + L)

debye_length = .83e-2
G = a / debye_length  # Normalised forcing width
np_potential = normalised_plasma_potential(a, debye_length, mass_ratio_o)

print("Plotting.")

f = forcing_space_debris(np_potential, xi, 1, vd, G)

plt.figure(figsize=(8, 6))
plt.scatter(xi, f)
plt.xlabel('x')
plt.ylabel('f')
plt.title('Forcing term for the normalised forced KdV equation')
plt.show()



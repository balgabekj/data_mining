import numpy as np
import matplotlib.pyplot as plt


# Define the function for Standing's correlation
def standing_correlation(p, api, T, yg):
    # Calculate the x parameter based on API gravity and temperature
    x = 0.0125 * api - 0.00091 * (T - 460)

    # Calculate Rs using the formula
    Rs = yg * ((p / 18.2 + 1.4) * 10 ** x) ** 1.2048

    return Rs


# Example values for the input parameters
p_values = np.linspace(100, 5000, 100)  # Pressure values in psia
api = 35  # Example API gravity
T = 520  # Temperature in Rankine (°R)
yg = 0.8  # Solution gas specific gravity

# Calculate Rs values for the range of pressure values
Rs_values = standing_correlation(p_values, api, T, yg)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(p_values, Rs_values, label=f'API = {api}, T = {T}°R, γg = {yg}')
plt.title("Standing's Correlation for Gas Solubility (Rs) vs Pressure (p)")
plt.xlabel("Pressure (psia)")
plt.ylabel("Gas Solubility Rs (scf/STB)")
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the function for y'
def y_prime(x):
    return (-1.169 * x / 7298**2) + (1.2731 / 7298)

# Generate x values
x_values = np.linspace(0, 10000, 1000)

# Calculate corresponding y' values
y_prime_values = y_prime(x_values)

# Create the plot
plt.plot(x_values, y_prime_values, label="y' = -1.169 * x / 7298^2 + 1.2731 / 7298")
plt.xlabel('x')
plt.ylabel("y'")
plt.title("Plot of y' = -1.169 * x / 7298^2 + 1.2731 / 7298")
plt.grid(True)
plt.legend()
plt.show()

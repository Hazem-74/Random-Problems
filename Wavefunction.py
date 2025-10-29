import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Define symbols
ca, a, b, k, x, w, t = sp.symbols('ca a b k x w t', real=True)
i = sp.I  # imaginary unit
pi = sp.pi

# Define the integrand
integrand = (ca / sp.sqrt(40 * pi)) * sp.exp(-a * (40 * k - b)) * sp.exp(-i * (40 * k * x + w * t))

# Perform the integration
integral_result = sp.integrate(integrand, (k, -sp.oo, sp.oo))

# Define constants for the plot
a_val = 1
b_val = 0
x_val = 1
w_val = 1
t_val = 1
ca_val = 1 / np.sqrt(40 * np.pi)

# Define the integrand as a function of k
def integrand_func(k):
    return ca_val * np.exp(-a_val * (40 * k - b_val)) * np.exp(-1j * (40 * k * x_val + w_val * t_val))

# Generate values for k
k_values = np.linspace(-10, 10, 400)
integrand_values = integrand_func(k_values)

# Plot the real part and imaginary part of the integrand
plt.figure(figsize=(10, 6))
plt.plot(k_values, integrand_values.real, label='Real part')
plt.plot(k_values, integrand_values.imag, label='Imaginary part', linestyle='--')
plt.title('Integrand as a function of k')
plt.xlabel('k')
plt.ylabel('Integrand')
plt.legend()
plt.grid(True)
plt.show()

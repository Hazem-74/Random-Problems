import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def func1(x):
    return (5 - x) * np.exp(x)

def func2(x):
    return np.full_like(x, 5)

# Generate x values
x = np.linspace(-2, 5, 1000)

# Calculate y values for y1 = (5 - x) * e^x
y1 = func1(x)

# Calculate y values for y2 = 5
y2 = func2(x)

# Find the intersected points
intersect_x = []
intersect_y = []
for i in range(len(x) - 1):
    if np.sign(y1[i] - y2[i]) != np.sign(y1[i + 1] - y2[i + 1]):
        intersect_x.append(x[i])
        intersect_y.append(y1[i])

# Plot the functions
plt.plot(x, y1, label='(5 - x) * e^x', color='blue')
plt.plot(x, y2, label='y = 5', color='green')
plt.scatter(intersect_x, intersect_y, color='red', label='Intersected Points')

# Customize the plot
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axvline(x=0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intersection of Functions')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
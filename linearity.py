import numpy as np
import matplotlib.pyplot as plt

# Generate linear data
x_linear = np.linspace(0, 10, num=50)
y_linear = 2*x_linear + 3 + np.random.normal(size=50)

# Generate non-linear data
x_nonlinear = np.linspace(0, 10, num=50)
y_nonlinear = 5*np.sin(x_nonlinear) + np.random.normal(size=50)

# Fit linear and non-linear models
coefficients_linear = np.polyfit(x_linear, y_linear, 1)
y_linear_fit = np.polyval(coefficients_linear, x_linear)

coefficients_nonlinear = np.polyfit(x_nonlinear, y_nonlinear, 5)
y_nonlinear_fit = np.polyval(coefficients_nonlinear, x_nonlinear)

# Visualize data and models
plt.figure(figsize=(8, 6))
plt.scatter(x_linear, y_linear, color='blue', label='Linear Data')
plt.plot(x_linear, y_linear_fit, color='red', label='Linear Fit')

plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Non-linear Data')
plt.plot(x_nonlinear, y_nonlinear_fit, color='purple', label='Non-linear Fit')

plt.legend()
plt.xlabel('Input Feature')
plt.ylabel('Target Variable')
plt.show()

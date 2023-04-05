import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Generate some noisy sinusoidal data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16))

# Fit a decision tree with max depth 2
tree2 = DecisionTreeRegressor(max_depth=2)
tree2.fit(X, y)

# Fit a decision tree with max depth 6
tree6 = DecisionTreeRegressor(max_depth=6)
tree6.fit(X, y)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X, tree2.predict(X), color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X, tree6.predict(X), color="yellowgreen", label="max_depth=6", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

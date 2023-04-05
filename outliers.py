import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

# generate a dataset with 100 samples and 1 feature
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# introduce 10 outliers in the dataset
np.random.seed(42)
outliers_index = np.random.choice(range(len(y)), size=10, replace=False)
y[outliers_index] += 50 * np.random.randn(10)

# fit a decision tree model to the data
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X, y)

# plot the decision tree model
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='data')
plt.scatter(X[outliers_index], y[outliers_index], color='red', label='outliers')
plt.plot(X, tree_model.predict(X), color='green', label='decision tree')
plt.title('Decision tree with outliers')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

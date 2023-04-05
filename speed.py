import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# generate a dataset with 10000 samples and 20 features
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)

# fit a decision tree model to the data and measure the time taken
start_time = time.time()
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X, y)
tree_time = time.time() - start_time
print("Time taken to fit decision tree model: {:.2f} seconds".format(tree_time))

# fit a linear model to the data and measure the time taken
start_time = time.time()
linear_model = LogisticRegression(random_state=42)
linear_model.fit(X, y)
linear_time = time.time() - start_time
print("Time taken to fit linear model: {:.2f} seconds".format(linear_time))

# plot the time taken for each model
models = ['Decision Tree', 'Linear Model']
times = [tree_time, linear_time]
colors = ['blue', 'green']
plt.bar(models, times, color=colors)
plt.title('Time taken to fit each model')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.show()

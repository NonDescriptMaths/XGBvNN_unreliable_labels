import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from query import shannon_entropy
import numpy as np

# Generate a binary classification dataset with 2-dimensional covariates
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=1)

# Split the data into labelled and unlabelled
X_labelled = X[:100]
y_labelled = y[:100]
X_unlabelled = X[100:]

# Train a logistic regression model on the labelled data
model = LogisticRegression()
model.fit(X_labelled, y_labelled)

# Define a predict method for the model that returns (x, y, y_hat)
def predict(data):
    y_hat = model.predict_proba(data)[:, 1]
    return (data, np.zeros_like(y_hat), y_hat)

# Replace the model's predict method with our custom one
model.predict = predict

# Use the shannon_entropy function with this model and the unlabelled data
selected_indices = shannon_entropy(model, 50, X)

# Get the selected samples
selected_samples = X[selected_indices]

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Plot the points with maximum entropy in dark green
plt.scatter(selected_samples[:, 0], selected_samples[:, 1], color='darkgreen')

plt.show()
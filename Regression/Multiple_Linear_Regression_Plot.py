import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(42)
X1 = 2 * np.random.rand(100, 1)
X2 = 3 * np.random.rand(100, 1)
y = 4 + 2 * X1 + 3 * X2 + np.random.randn(100, 1)

# Combine the two input variables into a single feature matrix
X = np.c_[X1, X2]

# Fit the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Obtain coefficients
b0 = model.intercept_[0]
b1, b2 = model.coef_[0]

# Display coefficients
print(f'y = {b0} + {b1}*X1 + {b2}*X2')

# Plot the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='blue', marker='o')

# Plot the regression plane
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
X_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
y_pred_mesh = model.predict(X_mesh).reshape(x1_mesh.shape)

ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, color='r', alpha=0.5, label='Regression Plane')

# Set labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('Multiple Linear Regression')

# Show the plot
plt.show()

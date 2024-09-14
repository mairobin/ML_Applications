import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class BaseRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def _add_bias_column(self, X):
        return np.vstack((np.ones((X.shape[0],)), X.T)).T

    def fit(self, X, y):
        X = self._add_bias_column(X)
        self.weights = np.zeros((X.shape[1], 1))
        n_samples = y.size

        for i in range(self.n_iterations):
            y_pred = self._approximation(X)
            y_diff = y_pred - y
            gradient = (1 / n_samples) * np.dot(X.T, y_diff)
            self.weights -= self.learning_rate * gradient

    def _approximation(self, X):
        raise NotImplementedError

    def _predict(self):
        raise NotImplementedError

    def predict(self, X):
        X = self._add_bias_column(X)
        return self._predict(X)

class LinearRegression(BaseRegression):
    def _approximation(self, X):
        return np.dot(X, self.weights)

    def _predict(self, X):
        return np.dot(X, self.weights)

class LogisticRegression(BaseRegression):
    def _sigmoid(self, x):
        x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))

    def _approximation(self, X):
        linear_model = np.dot(X, self.weights)
        return self._sigmoid(linear_model)

    def _predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted = self._sigmoid(linear_model)
        # Threshold predictions to 0 and 1
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Testing
if __name__ == "__main__":
    # Load breast cancer dataset
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Normalize features (optional but can improve performance)
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    # Initialize and train logistic regression model
    regressor = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    regressor.fit(X_train, y_train)

    # Make predictions
    predictions = regressor.predict(X_test)

    # Calculate accuracy
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print("Logistic regression classification accuracy:", accuracy)

import numpy as np

class LogisticRegressionModel:
    """
    Simple implementation of Logistic Regression using
    gradient descent for binary classification.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Args:
            learning_rate : step size for gradient descent
            n_iters : number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss = []

    def _sigmoid(self, z):
        """Sigmoid activation function (numerically stable)."""
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        expz = np.exp(z[~pos])
        out[~pos] = expz / (1.0 + expz)
        return out

    def fit(self, X, y):
        """
        Train the logistic regression model.
        Args:
            X : features, shape (n_samples, n_features)
            y : labels {0,1}, shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        # Gradient descent
        for _ in range(self.n_iters):
            # Linear combination
            linear_model = X @ self.weights + self.bias
            # Apply sigmoid to get probabilities
            y_predicted = self._sigmoid(linear_model)

            # binary cross-entropy loss (with epsilon for numerical safety)
            eps = 1e-12
            loss = (-y * np.log(y_predicted + eps) - (1 - y) * np.log(1 - y_predicted + eps)).mean()
            self.loss_.append(loss)


            # Compute gradients
            dw = (1.0 / n_samples) * (X.T @ (y_predicted - y))
            db = (1.0 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predict probabilities for input samples.
        Args:
            X : features
        Returns:
            probabilities in [0,1]
        """
        X = np.asarray(X, dtype=float)
        linear_model = X @ self.weights + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels.
        Args:
            X : features
            threshold : probability cutoff (default=0.5)
        Returns:
            array of predictions {0,1}
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
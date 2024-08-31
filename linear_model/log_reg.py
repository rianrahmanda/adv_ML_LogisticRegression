import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store the number of features
        self.n_features_in_ = X.shape[1]

        # Standardize features
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        # Add intercept if required
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize parameters
        self.theta_ = np.zeros(X.shape[1])

        # Gradient Descent
        m = X.shape[0]
        for _ in range(self.n_iterations):
            z = np.dot(X, self.theta_)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta_ -= self.learning_rate * gradient

        return self

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Standardize features
        X = self.scaler_.transform(X)

        # Add intercept if required
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute probabilities
        z = np.dot(X, self.theta_)
        probas = self._sigmoid(z)
        return np.vstack([1-probas, probas]).T

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

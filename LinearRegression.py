class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.w = None  # Initialize weights

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]  # Reshape if single feature
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(N)])  # Add bias column

        # Use pseudoinverse instead of inverse for stability
        self.w = np.linalg.pinv(x.T @ x) @ x.T @ y
        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]  # Reshape if single feature
        if self.add_bias:
            x = np.column_stack([x, np.ones(x.shape[0])])  # Add bias column
        return x @ self.w  # Predict y values

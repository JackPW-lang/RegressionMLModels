logistic = lambda z: 1./ (1 + np.exp(-z))       #logistic function

class LogisticRegression:

    def __init__(self, add_bias=True, learning_rate=0.05, epsilon=1e-3, max_iters=10000, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose

    def gradient(self, x, y):
        #Includes print statements for debugging
        y = y.ravel()
        N,D = x.shape
        yh = logistic(np.dot(x, self.w))    # predictions  size N
        #print(f"Prediction shape: {yh.shape}")
        grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
        #print(f"y shape: {y.shape}")
        #print(f"Gradient shape: {grad.shape}")
        return grad

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        self.w = np.zeros(D)
        g = np.inf
        t = 0
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            g = self.gradient(x, y)
            self.w = self.w - self.learning_rate * g
            t += 1

        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
        return self

    def predict_prob(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            #predict output
        return yh

    def predict(self, x, threshold=0.5):
      return self.predict_prob(x) >= threshold

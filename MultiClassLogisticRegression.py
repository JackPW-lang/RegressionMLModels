#Multiclass Logistic Regression
class Multinomial_logistic:
  def __init__(self, nFeatures, nClasses):
    self.W = np.random.rand(nFeatures, nClasses) / 100
    self.D = int(nFeatures)
    self.C = int(nClasses)

  def predict(self, X):
    y_pred = np.exp(np.matmul(X, self.W)) #Overflows occuring is handled by python
    sum_pred = y_pred.sum(axis=1).reshape(X.shape[0], 1)

    if (sum_pred.any(0)): #Ensure all divisions are valid
      sum_pred[sum_pred == 0] = 1

    #Ensure no NaN values
    if np.isnan(y_pred).any():
      print('nan in y_pred')
    if np.isnan(sum_pred).any():
      print('nan in sum_pred')

    #NaN occurs here
    temp = y_pred / (sum_pred + 1e-5)
    if np.isnan(temp).any():
      temp[np.isnan(temp)] = 1
    return temp

  def grad(self, X, y):
      gradient= np.matmul(X.transpose(), self.predict(X) - y)
      if np.isnan(gradient).any(): #Replacing NaN with zeros
        gradient[np.isnan(gradient)] = -1e-5

      return gradient

  def ce(self, X, y):
        pred = self.predict(X)
        temp = np.log(pred)

        if np.isnan(y).any():
            print('nan in y')
        if np.isnan(temp).any():
            print('nan in temp')

        y_temp = y * temp
        if np.isnan(y_temp).any(): #More validation for NaN values
            y_temp[np.isnan(y_temp)] = 1

        return -np.sum(y_temp)

    #n_stop will stop the descent when the past n_stop average CE is smaller than the new one
  def fit(self, X, y, X_valid=None, y_valid=None, lr=0.005, niter=100, to_print=False, n_stop = 1000):
        losses_train = np.zeros(niter)
        losses_valid = np.zeros(niter)
        maxiter = 0
        for i in range(niter):
            maxiter = maxiter + 1
            self.W = self.W - lr * self.grad(X, y)
            loss_train = self.ce(X, y)
            losses_train[i] = loss_train
            if X_valid is not None and y_valid is not None:
                loss_valid = self.ce(X_valid, y_valid)
                losses_valid[i] = loss_valid
                if (i > n_stop) and ( losses_valid[i - (n_stop + 1) : i - 1].sum() / n_stop < losses_valid[i] ):
                  print(f"Average Validation CE increased after iter {i}: ")
                  break
                if(to_print):
                  print(f"iter {i}: {loss_train:.3f}; {loss_valid:.3f}")
            else:
                if(to_print):
                  print(f"iter {i}: {loss_train:.3f}")
        return maxiter, losses_train, losses_valid

  def check_grad(self, X, y):
        N, C = y.shape
        D = X.shape[1]

        diff = np.zeros((D, C))

        W = self.W.copy()
        numeric_grad = np.zeros((D, C))
        derived_grad = np.zeros((D, C))
        for i in range(D):
            for j in range(C):
                epsilon = np.zeros((D, C))
                epsilon[i, j] = np.random.rand() * 1e-4

                self.W = self.W + epsilon
                J1 = self.ce(X, y)
                self.W = W

                self.W = self.W - epsilon
                J2 = self.ce(X, y)
                self.W = W

                numeric_grad[i,j] = (J1 - J2) / (2 * epsilon[i, j])
                derived_grad[i,j] = self.grad(X, y)[i, j]


                diff[i, j] = np.square(derived_grad[i,j] - numeric_grad[i,j]).sum() / \
                             np.square(derived_grad[i,j] + numeric_grad[i,j]).sum()

        print(f'Average Analytical gradient of all features and classes: {derived_grad.sum() / D*C}')
        print(f'Average Numerical gradient of all features and classes: {numeric_grad.sum() / D*C}')
        print(f'Sum of the squared difference over all features and classes: {diff.sum()}')

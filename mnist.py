import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
#Testing for data type and null values
print(test.dtypes.value_counts())
print(train.dtypes.value_counts())
print(train.isna().sum().sum())
print(test.isna().sum().sum())

class KNNClassifier():

    def fit(self, X, y):
        self.X = X
        self.y = y.astype(int)

    def predict(self, X, K, epsilon=1e-3):
        N = len(X)
        y_hat = np.zeros(N)
        # Calculating distances 
        for i in range(N):
            dist2 = np.sum((self.X-X[i])**2, axis=1)  # List of distances
            idxtK = np.argsort(dist2)[:K]    # Indicies in order up to K
            gamma = 1/(np.sqrt(dist2[idxtK]+epsilon))
            y_hat[i] = np.bincount(self.y[idxtK], weights=gamma).argmax()
        return y_hat
def acc(y, y_hat):
  return np.mean(y==y_hat)

#define training data into numpy array
train_X=train.iloc[:,3:].to_numpy()
train_y=train.iloc[:,2].to_numpy()
#define test data into numpy array
test_X=test.iloc[:,3:].to_numpy()
test_y=test.iloc[:,2].to_numpy()

#instantiate KNN classifier
knn = KNNClassifier()
#Fit training data
knn.fit(train_X,train_y)
#test test data
test_yhat = knn.predict(test_X, K=10)

#compare predicted results to actual results to get accuracy
result = acc(test_y, test_yhat)
print(result)



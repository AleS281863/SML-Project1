"""
=====================================================
Gaussian process classification with regression (GPC) 
=====================================================

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import LabelBinarizer

#choose a seed
seed = 2311
np.random.seed(seed)

# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape

#trainNumb = [1697] 
trainNumb = [20,50,100,500,1000,1500,1697]

for num in trainNumb:

    #N = 600
    Ntrain = num
    Ntest = 100


    Xtrain = X[0:Ntrain,:]
    ytrain = y[0:Ntrain]
    Xtest = X[N-Ntest:N,:]
    ytest = y[N-Ntest:N]

    #Using One Hot Encoding on the training data.
    #Notice we are only fitting the LB object on the training set (In our case nothing changes, but with some
    #preprocessing operations, including the test data might lead to overfitting)
    lb = LabelBinarizer()
    lb.fit(ytrain)
    ytrain_lb = lb.transform(ytrain)

    kernel = 1.0 * RBF([1.0]) # isotropic kernel
    #kernel = DotProduct(1.0) 
    #kernel = Matern(1.0)

    #Now we use a GaussianProcessRegressor object with output in R^K
    gpc_rbf = GaussianProcessRegressor(kernel=kernel).fit(Xtrain, ytrain_lb)
    
    #Prediction for the training data in R^K (This array has dimension Ntrain x K)
    yp_train = gpc_rbf.predict(Xtrain)

    #We select the most likely class by taking the maximal argument (so we go back to Ntrain X 1 )
    yp_train_results = np.argmax(yp_train,axis = 1)

    #Hinge Loss for training
    train_error_rate = np.mean(np.not_equal(yp_train_results,ytrain))

    #We do the same thing with the test dataset. Notice that we never need to encode it, 
    # as we can simply "decode" the output and compare it with the original ytest
    yp_test = gpc_rbf.predict(Xtest)
    yp_test_results = np.argmax(yp_test,axis = 1)
    test_error_rate = np.mean(np.not_equal(yp_test_results,ytest))


    print(f'Training error rate for training size of {num}')
    print(train_error_rate)
    print(f'Test error rate for training size of {num}')
    print(test_error_rate)
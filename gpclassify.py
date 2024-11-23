"""
=====================================================
Gaussian process classification (GPC) 
=====================================================

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic

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


    Xtrain = X[0:Ntrain-1,:]
    ytrain = y[0:Ntrain-1]
    Xtest = X[N-Ntest:N,:]
    ytest = y[N-Ntest:N]


    kernel = RBF(1.0) # isotropic kernel
    #kernel = DotProduct(1.0) 
    #kernel = Matern(1.0)
    #kernel = RationalQuadratic(1.0)

    #GaussianProcessClassifier performs hyperparameter estimation, this means that the the value specified above may not be the final hyperparameters
    #If you dont want it to do hyperparameter optimization set optimizer = None like this GaussianProcessClassifier(kernel=kernel,optimizer = None).fit(Xtrain, ytrain)
    #You can check the final hyperparameters with gpc_rbf.kernel_.get_params()['kernels']
    gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
    yp_train = gpc_rbf.predict(Xtrain)
    train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
    yp_test = gpc_rbf.predict(Xtest)
    test_error_rate = np.mean(np.not_equal(yp_test,ytest))
    print(f'Training error rate for training size of {num}')
    print(train_error_rate)
    print(f'Test error rate for training size of {num}')
    print(test_error_rate)



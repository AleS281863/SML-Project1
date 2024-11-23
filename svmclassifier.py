import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct

# import some data to play with
digits = datasets.load_digits()


#choose a seed
seed = 2311
np.random.seed(seed)
# Load data into train set and test set
digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape

#trainNumb = [1697] 
trainNumb = [20,50,100,500,1000,1500,1697]
#alphaNum = [0.5] 
alphaNum = [1,0.75,0.5,0.33,0.25,0.125,0.1]


for num in trainNumb:
    for alpha in alphaNum:
        

        Ntest = int(100)
        Ntrain = num
        Xtrain = X[0:Ntrain,:]
        ytrain = y[0:Ntrain]
        Xtest = X[N-Ntest:N,:]
        ytest = y[N-Ntest:N]


        def svmsubgradient(Theta, x, y):
        #  Returns a subgradient of the objective empirical hinge loss
        #
        # The inputs are Theta, of size n-by-K, where K is the number of classes,
        # x of size n, and y an integer in {0, 1, ..., 9}.
            G = np.zeros(Theta.shape)
            ## IMPLEMENT THE SUBGRADIENT CALCULATION -- YOUR CODE HERE

            ## We are working on a single data point. This could easily be generalized to multiple data points by
            ## using a for loop on all data points and then averaging the subgradients.

            ## Compute maximizing column index
            j = 0
            jStar = None
            
            #This checks that L(Theta) > 0
            maxL = 0
            
            while j < Theta.shape[1]:
                #Computing Lj
                Lj = 1 + np.dot(x,Theta[:,j]-Theta[:,y])

                #This is also checking if Lj is greater or equal to 0.
                #The >= means we are also giving a non zero subgradient if we are on the edge of the subspace where
                #the function is flat. Not sure if this is good or bad but we will test it. (Lj being less than zero
                #for all Lj means we are inside of the flat subspace. If one of them is exactly zero we are on its edge 
                #and jStar is one of the columns where we are on such edge). Of course any positive Lj goes over all of this.
                if Lj >= maxL:
                    jStar = j
                    maxL = Lj
                j = j+1

            #Computing G (We will only compute it if we found a jStar such that maxL >= 0. As mentioned before, this means
            # we are providing a subgradient even if maxL = 0 because one or more Lj's are exactly zero)
            if  jStar != None:
                G[:,jStar] = x
                G[:,y] = -x
            return(G)

        def sgd(Xtrain, ytrain, maxiter = 10, init_stepsize = 1.0, l2_radius = 10000):
        #
        # Performs maxiter iterations of projected stochastic gradient descent
        # on the data contained in the matrix Xtrain, of size n-by-d, where n
        # is the sample size and d is the dimension, and the label vector
        # ytrain of integers in {0, 1, ..., 9}. Returns two d-by-10
        # classification matrices Theta and mean_Theta, where the first is the final
        # point of SGD and the second is the mean of all the iterates of SGD.
        #
        # Each iteration consists of choosing a random index from n and the
        # associated data point in X, taking a subgradient step for the
        # multiclass SVM objective, and projecting onto the Euclidean ball
        # The stepsize is init_stepsize / sqrt(iteration).
            K = 10
            NN, dd = Xtrain.shape
            #print(NN)
            Theta = np.zeros(dd*K)
            Theta.shape = dd,K
            mean_Theta = np.zeros(dd*K)
            mean_Theta.shape = dd,K
            r = l2_radius #Radius of the Euclidean ball
            ## YOUR CODE HERE -- IMPLEMENT PROJECTED STOCHASTIC GRADIENT DESCENT

            iter = 1
            notConverged = True
            while notConverged == True:
                
                #Compute stepsize
                stepsize = init_stepsize/iter**(alpha)
                
                #Choose random index
                index = np.random.randint(0,NN)
                
                #Compute stochastic subgradient
                G = svmsubgradient(Theta, Xtrain[index,:], ytrain[index])
                #if iter % 1000 == 0:  # Print every 1000 iterations
                #    print(f"Iteration {iter}, Gradient norm: {np.linalg.norm(G)}")
                #    print(f"Iteration {iter}, Loss: {compute_loss(Xtrain, ytrain, Theta)}")

                #Compute ThetaOut. This might be outside the Euclidean ball, so we compute its Frobenius norm
                #and if necessary we project it onto the r-Ball
                ThetaOut = Theta - stepsize*G
                ThetaNorm = np.linalg.norm(ThetaOut,'fro')
                if ThetaNorm <= r:
                    Theta = ThetaOut
                else:
                    Theta = ThetaOut * r/ThetaNorm 


                mean_Theta = mean_Theta + Theta
                iter = iter + 1

                #Check for convergence -- Implemented like this because it could be extended into a different convergence criterion
                #Even if we were asked to perform a certain number of iterations
                if iter >= maxiter:
                    notConverged = False
            mean_Theta = 1/iter * mean_Theta
            return Theta, mean_Theta

        def compute_loss(X, y, Theta):
            loss = 0
            for i in range(X.shape[0]):
                margins = 1 + np.dot(X[i], Theta) - np.dot(X[i], Theta[:, y[i]])
                margins[y[i]] = 0  # Exclude the correct class
                loss += max(0, np.max(margins))
            return loss / X.shape[0]

        def Classify(Xdata, Theta):
        #
        # Takes in an N-by-d data matrix Adata, where d is the dimension and N
        # is the sample size, and a classifier X, which is of size d-by-K,
        # where K is the number of classes.
        #
        # Returns a vector of length N consisting of the predicted digits in
        # the classes.
            scores = np.matmul(Xdata, Theta)
            #print(scores)
            inds = np.argmax(scores, axis = 1)
            return(inds)



        l2_radius = 40.0
        M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
        init_stepsize = l2_radius/M_raw
        maxiter = 40000
        Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius)
        print(f'Error rate with nTrain = {num} and alpha = {alpha}')
        results = Classify(Xtest, mean_Theta)
        print(np.sum(np.not_equal(results,ytest)/Ntest))



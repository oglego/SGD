Description: In this program we will implement and evaluate the
gradient descent algorithm and apply it to a linear regression
problem.  

Our linear regression model will be implemented by using the below steps:
    
a) we will first generate our data by using the following equation:
        
    y = Xw + b + e
    
   where y is our target value, X is our feature matrix, w will be
   the weights vector corresponding to 3 features, b is the bias, and
   e is the noise added to the data.  We will assume that the noise
   follows a Gaussian distribution with zero mean and sigma standard
   deviation.  The dataset that we generate will be stored in a CSV 
   file titled "data.csv".
   
b) next we will implement the gradient descent algorithm.  We will use
   the mean squared loss function as our loss function.  The gradient
   function will be defined and implemented in order to optimize our 
   hypothesis for the weights and bias.  Our implementation of the
   gradient descent algorithm will follow the mini-batch stochastic
   gradient descent with batch size as one of the input parameters.
   When the batch size is equal to the number of training examples,
   the gradient descent is batch gradient descent.  When the batch
   size is 1, gradient descent is stochastic gradient descent.

Note that to run the program you will need the below libraries installed:

numpy
random
matplotlib
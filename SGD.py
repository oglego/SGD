# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:06:31 2022

Author: Aaron M Ogle

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
   
"""
import numpy as np # Import the numpy library for linear algebra operations
import random # Used in stochastic gradient descent
import matplotlib.pyplot as plt # used for plotting

"""
-----------------------------------------------------------------------------
Function - generate_data

Parameters - N/A
-----------------------------------------------------------------------------

The function generate_data will use the following equation

y = Xw + b + e

to generate data for us to use for our implementation
of the gradient descent algorithm.  For further information
on this equation, please see section "a" in our above description.

The function will return data that can be used for testing, but the
main goal of the function is to create a csv that houses the data
we will be using to analyze the gradient descent algorithms.
"""

def generate_data():
    # Create a matrix that contains 1000 rows with 3 columns
    # where the values in the matrix are integers in the set [0,10]
    X = np.random.randint(10, size = (1000,3))
    # Define a vector for our weights - 
    w = [3, 2.5, -1]
    # Redefine our weight list as a numpy array
    w = np.asarray(w)
    # Define a scalar for our bias
    b = 1
    # Set our noise to follow a gaussian distribution with zero
    # mean and a standard deviation of 0.03
    e = np.random.normal(0, 0.03)
    # Denote variables for the size of our matrix
    rows = 1000
    cols = 3
    # Create an empty list to store our y values
    y = []
    # Compute our y values
    for row in range(rows):
        # We need to sum the X values and weights for
        # each row but we want to reset our summation
        # as we move through each row so that we can
        # restart the sum at the next row
        summ = 0
        for col in range(cols):
            # Sum the values X multiplied by the weights
            summ += X[row][col] * w[col]
        # Append the value of adding in the bias and error to
        # to our summation and add it to our list of y values
        y.append(summ + b + e)
    # Convert our y list into a numpy array
    y = np.asarray(y)
    # Reshape our y vector
    shape = (1000,1)
    y = y.reshape(shape)
    # Combine our X values and y values into one array so
    # that we can export it to a cvs
    data = np.append(X,y,1)
    # Put our data into a csv
    np.savetxt("data.csv", data, delimiter=",")
    # Return the values of X[0], w, b, e, and y[0] for testing
    return X[0], w, b, e, y[0]

"""
-----------------------------------------------------------------------------
Function - gradient_descent

Parameters - 

data: The data parameter will contain a matrix X with a corresponding
target vector y.  The data itself is a matrix that contains the y vector
in the last column.  We will parse the data out in the function itself.

weights: We will experiment with
different weight values - the values for our weights will first be
all zero, then all uniform random, and finally following a normal
distribution.  We pass the weights as a parameter into our function
so that we can experiment with diffent values.

epochs (iterations) - We will pass a value for epochs (iterations)
into the function so that we can experiment with varying the number
of iterations to see how the initial weights and bias should be set.

batch size - The gradient descent function will change depending on
what batch size is passed as a parameter.  For example, if a subset
of the samples are used as the batch size then the algorithm will
become the mini batch stochastic gradient descent algorithm.  
If the batch size is equal to the total number of training samples then 
the algorithm will be the batch gradient descent algorithm.  If the 
batch size is equal to one then it is the stochastic gradient
descent algorithm.

n (learning rate) - we will pass the learning rate to our function 
where the learning rate is a notion of how "big" or "small" of a step
to take in gradient descent.

bias - we will pass in a value for a bias so that we can add in a
notion of noise to our data.

-----------------------------------------------------------------------------

The function gradient descent will implement the gradient descent 
optimization algorithm.  The gradient descent optimization algorithm 
is used to find the local or global minimas in differentiable
functions.
"""

def gradient_descent(data, weights, epochs, batch_size, n, bias):
    # Our data is in a file so we want to read the data from
    # the csv file and input it into a numpy array
    data = np.loadtxt(open("data.csv", "rb"), delimiter=",")
    # Parse out the X and y values from our data set into their
    # own respective matrix and vector
    # Below syntax will use slicing to get all of the columns
    # in our dataset except the last column, this is our matrix
    # of X values
    X = data[:, :-1]
    # Use slicing to get the y values from our input dataset
    y = data[:,-1]
    # Reshape y so that it is a vector of 1000 rows by 1 column
    shape = (1000, 1)
    y = y.reshape(shape)
    # Now that we have parsed out the data we will use the
    # gradient descent algorithm on it
    # We will need to calculate the mean square error so
    # we create a variable for it and initialize it to zero
    mse = 0
    # Create an empty list to store the value of our
    # mean square error results - this will be used for
    # plotting later
    mse_values = []
    # Create an empty list for our epochs values
    # this will be used for plotting later
    epoch_values = []
    # Create a variable m which denotes the length of the vector y
    m = len(y)
    # Iterate through the number of epochs that
    # were passed to the function
    for i in range(epochs):
        # Add the value of i to our epochs list
        epoch_values.append(i)
        # Iterate through our data depending on the batch size
        for j in range(batch_size):
            # Create and set variable J to 0 - J will be the value
            # for the gradient that is calculated. 
            J = 0
            # If the batch size is one then this is 
            # stochastic gradient descent so we want to
            # randomly select a row in the data set to avoid
            # using the same row through every epoch
            if batch_size == 1:
                # Randomly pick a row
                r = random.randint(0,len(y)-1)
                # Compute the gradient 
                J += (sum(X[r] * (weights)) - y[r]) * X[r] 
                # Update our weights based off subtracting
                # the learning rate multiplied by the gradient
                weights = weights - n*J
                # Compute the mean square error 
                mse = mse + ((y[r] - sum(X[r] * weights + bias))**2)
            # Same logic as above but instead we will be using
            # batch or mini batch gradient descent
            else:
                # Compute the gradient 
                J += (sum(X[j] * (weights)) - y[j]) * X[j] 
                # Update our weights based off subtracting
                # the learning rate multiplied by the gradient
                weights = weights - n*J
                # Compute the mean square error 
                mse = mse + ((y[j] - sum(X[j] * weights + bias))**2)
        # After summing up all of the mean square errors we need to
        # multiply it by the total number of rows and then add
        # it to our mean square error list so that we can plot it
        mse = (1/m) * mse
        mse_values.append(mse.astype('float32'))
        # Display the values for mean square error
        print(f"EPOCH {i+1} - MSE {mse}")
    # Iterate through our mse values and change the type,
    # after the type is changed add it to a new list for
    # our plot
    mse_vals_cleaned = []
    for i in mse_values:
        mse_vals_cleaned.append(round(float(i),2))
    # Plot the mean square error values for each epoch
    plt.plot(epoch_values, mse_vals_cleaned, color="green")
    plt.xlabel("Epoch (Iteration)")
    plt.ylabel("Mean Square Error")
    # Return the weights that were computed
    return weights

"""
The main function of our program will be our driver function which is
used to call all of the other functions in our program.
"""
def main():
    
    # Call our generate data function so that we can have data
    # to work with
    """
    The below generate_data function will be commented out and 
    we will use the file that was originated from the original
    run of the function; if you want to test this function
    simply uncomment it.
    
    Note that I will include the "data.csv" file that is used 
    in this program.
    """
    #data = generate_data()
    # Set the below variable to the name of the file that your
    # data is contained in
    data = "data.csv"
    
    """
    *****************************************************************************************
    Parameters for - How to initialize the weights and bias?
    *****************************************************************************************
    
    NOTE - in the below we will be testing different values for the weights so depending
    on which value for the weights you want to use you will need to comment/uncomment
    different sections.
    """
    """
    # Create a list of weights set to 0 for question a)
    #w = np.asarray([0, 0, 0])
    
    # Create a list of weights that are uniform random for question a)
    #w = np.asarray([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    
    # Create a list of weights that are from the normal distribution
    # with mean 0 and standard deviation 0.03
    
    w = np.asarray([np.random.normal(0, 0.03), np.random.normal(0, 0.03), np.random.normal(0, 0.03)])
    
    
    # Input the number of epochs (iterations) that you would
    # like for the algorithm to go through
    epochs = 200
    
    # Input the batch size that the algorithm will use
    # Recall that if the batch size is one then the algorithm
    # is stochastic gradient descent, if it is 1000 then it
    # is batch gradient descent, and if it is between 1 and
    # 1000 then it is mini batch gradient descent
    batch_size = 50
    
    # Set the learning rate, the learning rate determines
    # how "big" or "small" of a step to take in the gradient
    # descent algorithm
    n = 0.02
    
    # Set the bias for our data
    bias = .9
    
    # Call the gradient descent algorithm
    print(gradient_descent(data, w, epochs, batch_size, n, bias))
    """
    
    
    ########################################################################################################
    
    """
    *****************************************************************************************
    Parameters for - Comparing stochastic, batch, and mini-batch algorithms.
    *****************************************************************************************
    
    NOTE - in the below we will be testing different values for the batch size so you will
    need to update accordingly.
    """
    
    """
    # Create a list of weights that are uniform random 
    w = np.asarray([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
        
    # Input the number of epochs (iterations) that you would
    # like for the algorithm to go through
    epochs = 200
    
    # Input the batch size that the algorithm will use
    # Recall that if the batch size is one then the algorithm
    # is stochastic gradient descent, if it is 1000 then it
    # is batch gradient descent, and if it is between 1 and
    # 1000 then it is mini batch gradient descent
    batch_size = 1000
    
    # Set the learning rate, the learning rate determines
    # how "big" or "small" of a step to take in the gradient
    # descent algorithm
    n = 0.02
    
    # Set the bias for our data
    bias = .9
    
    # Call the gradient descent algorithm
    print(gradient_descent(data, w, epochs, batch_size, n, bias))
    """
    #########################################################################################################
    
    
    ########################################################################################################
    
    """
    *****************************************************************************************
    Parameters for - How does learning rate affect time to converge?
    *****************************************************************************************
    
    NOTE - in the below we will be testing different values for the learning rate so you will
    need to update accordingly.
    
    If the learning rate is >= 0.03, the algorithm fails as the step size for it becomes
    too large causing an overflow for the numpy arrays
    """
    
    # Create a list of weights that are uniform random for question 
    w = np.asarray([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
        
    # Input the number of epochs (iterations) that you would
    # like for the algorithm to go through
    epochs = 200
    
    # Input the batch size that the algorithm will use
    # Recall that if the batch size is one then the algorithm
    # is stochastic gradient descent, if it is 1000 then it
    # is batch gradient descent, and if it is between 1 and
    # 1000 then it is mini batch gradient descent
    batch_size = 50
    
    # Set the learning rate, the learning rate determines
    # how "big" or "small" of a step to take in the gradient
    # descent algorithm
    n = 0.0001
    
    # Set the bias for our data
    bias = .9
    
    # Call the gradient descent algorithm
    print(gradient_descent(data, w, epochs, batch_size, n, bias))
    
    #########################################################################################################
    
main()


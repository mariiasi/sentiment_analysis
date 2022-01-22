import numpy as np
from src.utils import sigmoid

def gradientDescent(x, y, w, b, learning_rate, num_iters, print_cost=False):
    '''
    *This module is test via unit test in tests

    A simple logistic regression model

    Arguments:
        x -- matrix of features which is (nx, m)
        y -- corresponding labels of the input matrix x, dimensions (1, m)
        w -- weights, a numpy array of size (nx, 1) initialized to = 0
        b -- bias, a scalar
        learning_rate -- learning rate of the gradient descent update rule
        num_iters: number of iterations you want to train your model for
    Return:
        cost -- the final cost
        w -- learnt weights
        b -- learnt bias
        costs -- list of costs
    '''

    print("gradientDescent is running")
    # get 'm', the number of rows in matrix x
    nx, m = x.shape[0], x.shape[1]
    costs=[]

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(w.T, x) + b

        # compute activation
        a = sigmoid(z)

        # compute the cost function
        cost = - 1 / m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        if i % 100 == 0:
            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))
        costs.append(float(cost))
        # update the weights theta
        dw = 1 / m * np.dot(x, (a - y).T)
        db = 1 / m * np.sum(a - y)

        w = w - np.dot(learning_rate, dw)
        b = b - np.dot(learning_rate, db)

        assert w.shape == (nx, 1)


    cost = float(cost)
    return cost, w, b, costs
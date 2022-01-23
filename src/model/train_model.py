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

def naiveBayes(freqs, y):
    """
    Args:
        freqs -- a dictionary corresponding to the frequencies of each tuple (word, label)
        y -- corresponding labels of the input matrix x, dimensions (1, m)

    Returns:
        logprior -- the log prior, log(m_pos / m_neg)
        loglikelihood -- the log likelihood of you Naive bayes equation. (equation 6 above)
    """
    # V is the number of unique words in the vocabulary for all classes, whether positive or negative
    # set(a) - returns unique elements in a, where a is a dictionery/array/etc
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # compute N_pos and N_neg - total number of positive and negative words for all tweets
    N_pos, N_neg = 0, 0
    for pair in freqs.keys():
        # if the label is positive (1 > 0)
        if pair[1] > 0:
            N_pos += freqs[pair]
        # else, the label is negative
        else:
            N_neg += freqs[pair]

    #the total number of tweets
    m = y.shape[1]

    # the number of positive and negative tweets
    m_pos = np.sum(y[0, :])
    m_neg = m - m_pos

    # compute log_prior
    logprior = np.log(m_pos / m_neg)

    # compute log_likelihood
    loglikelihood = {}
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        # compute positive and negative probabilities of each word
        P_w_pos = (freq_pos + 1) / (N_pos + V)
        P_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(P_w_pos / P_w_neg)

    return logprior, loglikelihood
import numpy as np
from src.utils import sigmoid, process_tweet
from src.features.make_features import extract_features

def logisticRegression_predict(tweet, freqs, w, b):
    '''
    Arguments:
        tweet -- a string
        freqs -- a dictionary corresponding to the frequencies of each tuple (word, label)
        w -- weights, dimention (nx, 1)
        b -- bias, scalar
    Returns:
        y_pred -- the probability of a tweet being positive or negative
    '''

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(w.T, x) + b)

    return y_pred

def naiveBayes_predict(tweet, logprior, loglikelihood):
    """

    Arguments:
        tweet -- a string
        logprior -- log(m_pos/m_neg) accounts for imbalance in datasets
        loglikelihood -- a dictionary of words mapping to numbers

    Returns:
        p -- the sum of all the logliklihoods of each word in the tweet
             (if found in the dictionary) + logprior (a number)

    """
    words = process_tweet(tweet)

    # initialise probability to logprior
    p = logprior
    for word in words:
        # if word exists in loglikelihood dictionary, add the number, otherwise add zero
        p += loglikelihood.get((word), 0)

    return p
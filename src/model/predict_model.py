import numpy as np
from src.utils import sigmoid
from src.features.make_features import extract_features

def predict_tweet(tweet, freqs, w, b):
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
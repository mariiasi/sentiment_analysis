import numpy as np
from src.model.predict_model import logisticRegression_predict, naiveBayes_predict
from src.utils import process_tweet


def logisticRegression_evaluate(x, y, freqs, w, b, show_misclassifications=False):
    """
    Arguments:
        x -- a list of tweets
        y -- (m, 1) vector with the corresponding labels for the list of tweets
        freqs -- a dictionary with the frequency of each pair (or tuple)
        theta -- weight vector of dimension (3, 1)
    Returns:
        accuracy -- (# of tweets classified correctly) / (total # of tweets)
    """

    print("evaluate is running")
    # the list for storing predictions
    y_hat = []

    for tweet in x:
        # get the label prediction for the tweet
        y_pred = logisticRegression_predict(tweet, freqs, w, b)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = (np.asarray(y_hat) == np.squeeze(y)).sum() / len(x)

    if show_misclassifications == True:
        print("Misclassifications:")
        print('Truth Predicted')
        for i in range(len(y_hat)):
            if np.abs(y[0, i] - (y_hat[i] > 0.5)) > 0:
                print('%d\t%0.1f\t%s' % (y[0, i], y_hat[i], ' '.join(process_tweet(x[i])).encode('ascii', 'ignore')))

    return accuracy

def naiveBayes_evaluate(x, y, logprior, loglikelihood, show_misclassifications=False):
    """
    Arguments:
        x -- a list of tweets
        y -- (m, 1) vector with the corresponding labels for the list of tweets
        freqs -- a dictionary with the frequency of each pair (or tuple)
        theta -- weight vector of dimension (3, 1)
    Returns:
        accuracy -- (# of tweets classified correctly) / (total # of tweets)
    """

    print("evaluate is running")
    # the list for storing predictions
    y_hat = []

    for tweet in x:
        # get the label prediction for the tweet
        y_pred = naiveBayes_predict(tweet, logprior, loglikelihood)
        if y_pred > 0.0:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = (np.asarray(y_hat) == np.squeeze(y)).sum() / len(x)

    if show_misclassifications == True:
        print("Misclassifications:")
        print('Truth Predicted')
        for i in range(len(y_hat)):
            if np.abs(y[0, i] - (y_hat[i] > 0.5)) > 0:
                print('%d\t%0.1f\t%s' % (y[0, i], y_hat[i], ' '.join(process_tweet(x[i])).encode('ascii', 'ignore')))

    return accuracy
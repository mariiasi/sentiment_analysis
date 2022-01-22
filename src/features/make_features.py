import numpy as np
from src.utils import process_tweet

def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    This function takes in a single tweet, processes it, loops through each word in the list of processed words,
    returns a feature vector x = (x0, x1, x2) = (1, freqs for positive word, freq for negative word)

    Arguments:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Returns:
        x: a feature vector of dimension (2, 1)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    words = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((2, 1))

    # bias term is set to 1
    #x[0, 0] = 1

    # loop through each word in the list of words
    for word in words:
        # increment the word count for the positive label 1
        x[0, 0] += freqs.get((word, 1), 0)  # if a word with label=1 does not exists in the dictionary => add 0,
                                            # othrtwise add its frequency

        # increment the word count for the negative label 0
        x[1, 0] += freqs.get((word, 0), 0)  # if a word with label=0 does not exists in the dictionary => add 0,
                                            # othrtwise add its frequency

    assert (x.shape == (2, 1))
    return x

if __name__ == '__main__':
    tweet = "I love to work from home :)"
    freqs = {('love', 1.0): 336, ('love', 0.0): 128,
             ('work', 1.0): 89, ('work', 0.0): 102,
             ('home', 1.0): 20, ('home', 0.0): 52,
             (':)', 1.0): 2960, (':)', 0.0): 2}
    x = extract_features(tweet, freqs, process_tweet=process_tweet)
    assert x[0, 0] == 3405 and x[1, 0] == 284
    print("\033[92m A simple test went well! :) \033[39m")
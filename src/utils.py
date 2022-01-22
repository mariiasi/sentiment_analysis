import re
from os import path
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    """
    Process tweet function.

    Arguments:
        tweet -- a string containing a tweet
    Returns:
        tweets_clean -- a list of words containing the processed tweet.

    """

    # two levels up: path.abspath(path.join(__file__, "../.."))
    filePath = path.abspath(__file__ + "/../../") + '/nltk_data'
    nltk.data.path.append(filePath)

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)

    # remove hashtags by only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords.words('english') and     # remove stopwords in English
                word not in string.punctuation):           # remove punctuation
            stem_word = PorterStemmer().stem(word)  # stemming word
            tweet_clean.append(stem_word)

    return tweet_clean

def build_freqs(tweets, labels):
    """
    Creates a dictionary mapping each (PROCESSED word, sentiment) pair to its frequency, i.e.,
    how frequent a certain (PROCESSED word, sentiment) pair appears in the dictionary.

    Arguments:
        tweets -- a list of tweets
        ys -- an m x 1 array with the sentiment label of each tweet (either 0 or 1)
    Returns:
        freqs -- a dictionary mapping each (word, sentiment) pair to its frequency
    """

    print("build_freqs is running")
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    labels_list = np.squeeze(labels).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(labels_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def sigmoid(z):
    """
    Arguments:
        z -- A scalar or numpy array of any size.

    Return:
        h -- the sigmoid of z
    """

    h = 1 / (1 + np.exp(-z))
    return h

if __name__ == '__main__':
    tweet = "I love to work from home :)"
    print(process_tweet(tweet))
    tweets = ["I am happy to work work and work.", "I won't work."]
    labels = np.array([[1], [0]])
    assert build_freqs(tweets, labels) == {('happi', 1): 1, ('work', 1): 3, ('work', 0): 1}
    assert np.isclose(sigmoid(0.0), 0.5)
    print("\033[92m Simple tests went well! :) \033[39m")


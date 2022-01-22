import random
from os import path
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import twitter_samples

def preprocess_data(train_size=0.8, test_size=0.2, valid_size=0, printInfo=False):
    """
    This function splits labeled tweets into train/test sets of tweets.

    Arguments:
        train_size -- the fraction of the train set relative to the whole dataset

    Returns:
        train_x -- train set of raw tweets, data type is list
        train_y -- labels of the train set, data type is numpy array
        test_x -- test set of raw tweets, data type is list
        test_y -- labels of the test set, data type is numpy array
    """
    print("preprocess data is running")
    # three levels up: path.abspath(path.join(__file__, "../.."))
    filePath = path.abspath(__file__ + "/../../../") + '/nltk_data'
    nltk.data.path.append(filePath)

    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    assert len(all_positive_tweets) == len(all_negative_tweets), "The amount of positive and negative sets should be equal!"

    # Tran test split split the data into two pieces, one for training and one for testing (validation set)
    assert (train_size + test_size + valid_size) == 1.0, "Check train/test/valid sizes"
    len_train = round(train_size * len(all_positive_tweets))
    train_pos = all_positive_tweets[ : len_train]
    test_pos = all_positive_tweets[len_train : ]
    train_neg = all_negative_tweets[ : len_train]
    test_neg = all_negative_tweets[len_train : ]

    # combine positive and negative examples, type list
    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # combine positive and negative labels, type list
    train_y = np.append(np.ones((1, len(train_pos))), np.zeros((1, len(train_neg))), axis=1)
    test_y = np.append(np.ones((1, len(test_pos))), np.zeros((1, len(test_neg))), axis=1)

    if __name__ == '__main__' or printInfo == True:
        print("========= About the data: ============")

        # Print info data structure of the datasets and a few example tweets
        print('The type of all_positive_tweets is: ', type(all_positive_tweets))
        print('The type of a tweet entry is: ', type(all_negative_tweets[0]), '\n')
        # Print a report with the number of positive and negative tweets
        print('Number of positive tweets: ', len(all_positive_tweets))
        print('Number of negative tweets: ', len(all_negative_tweets))
        # Print a random positive tweet in greeen
        print("\nExample tweets:")
        print('Positive: ' + '\033[92m' + all_positive_tweets[random.randint(0, len(all_positive_tweets))] + '\033[39m')
        # Print a random negative tweet in red
        print('Negative: ' + '\033[91m' + all_negative_tweets[random.randint(0, len(all_negative_tweets))] + '\033[39m')

        # Visualize the report
        fig = plt.figure(1, figsize=(6, 3))
        plt.subplot(121)
        colors = ['green', 'red']
        labels = ['Positive \n tweets', 'Negative \n tweets']
        sizes = [len(all_positive_tweets), len(all_negative_tweets)]
        plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Dataset')
        plt.tight_layout()

        print('\nSize of train set, its type: ', len(train_x), type(train_x))
        print('Size of test set, its type: ', len(test_x), type(train_y))
        print("\nType of labels =", type(train_y))
        print("Shape of train labels = ", str(train_y.shape))
        print("Shape of test labels = ", str(test_y.shape))
        print('Positive label: ' + '\033[92m' + '1' + '\033[39m')
        print('Negative label: ' + '\033[91m' + '0' + '\033[39m')

        print('\nThis is an example of a tweet: \n', train_x[10])

        #Visualize the train test split
        plt.subplot(122)
        colors = ['green', 'red', 'green', 'red']
        labels = ['Positive\ntest', 'Negative\ntest', 'Positive\ntrain', 'Negative\ntrain']
        sizes = [len(test_pos), len(test_neg), len(train_pos), len(train_neg)]
        plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=145)
        plt.axis('equal')
        plt.title('Train test sets')
        plt.tight_layout()

        """# check the output
        print("\ntype(freqs) = " + str(type(freqs)))
        print("len(freqs) = " + str(len(freqs.keys())))
        # Print 10 most frequent PROCESSED words in the dictionary
        print("\n10 most frequent PROCESSED items in the dictionary:")
        count = 0
        for keys, values in zip(sorted(freqs, key=freqs.get, reverse=True), sorted(freqs.values(), reverse=True)):
            print(keys, values)
            count += 1
            if count == 10: break"""
        print("========= About the data END =========")
        plt.show()
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    preprocess_data(train_size=0.8, printInfo=True)
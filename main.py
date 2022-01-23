import numpy as np

from src.utils import build_freqs
from src.features.make_features import extract_features
from src.data.make_datasets import preprocess_data
from src.model.train_model import gradientDescent, naiveBayes
from src.model.evaluate_model import logisticRegression_evaluate, naiveBayes_evaluate

#process data
train_x, train_y, test_x, test_y = preprocess_data(train_size=0.8, test_size=0.2, valid_size=0.0)

#build freqs dictionary
freqs = build_freqs(train_x, train_y)

X = np.zeros((2, len(train_x)))
for i in range(len(train_x)):
    X[:, i] = np.squeeze(extract_features(train_x[i], freqs))
Cost, w, b, costs = gradientDescent(X, train_y, w=np.zeros((2, 1)), b=0.0, learning_rate=1e-9, num_iters=10000, print_cost=True)


train_accuracy = logisticRegression_evaluate(train_x, train_y, freqs, w, b)
print("Logistic Regression train_accuracy = ", train_accuracy)
test_accuracy = logisticRegression_evaluate(test_x, test_y, freqs, w, b)
print("Logistic Regression test_accuracy = ", test_accuracy)

# Naive Bayes
logprior, loglikelihood = naiveBayes(freqs, train_y)
train_accuracy = naiveBayes_evaluate(train_x, train_y, logprior, loglikelihood)
print("Naive Bayes train_accuracy = ", train_accuracy)
test_accuracy = naiveBayes_evaluate(test_x, test_y, logprior, loglikelihood,)
print("Naive Bayes test_accuracy = ", test_accuracy)

print("!!!!! Finished !!!!!!!!")


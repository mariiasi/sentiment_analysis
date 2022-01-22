import unittest

import numpy as np
from src.utils import sigmoid
from src.model.train_model import gradientDescent

class ActivationFunctionsTest(unittest.TestCase):
   def test_sigmoid(self):
       test_cases = [
           {
               "name": "zero_check",
               "input": 0.0,
               "expected": 0.5,
           },
           {
               "name": "negative_check",
               "input": -20.0,
               "expected": 2.0611536181902037e-09,
           },
           {
               "name": "positive_check",
               "input": 20.0,
               "expected": 0.9999999979388463,
           },
       ]
       for test_case in test_cases:
           result = sigmoid(test_case["input"])
           self.assertTrue(np.isclose(result, test_case["expected"]), "Wrong output from sigmoid function in test: " + test_case["name"])

class ModelTest(unittest.TestCase):
#gradientDescent(x, y, w, b, learning_rate, num_iters, print_cost=False)
    def test_gradientDescent(self):
        """
        x and y were generates as:
            m = 10
            x = np.random.uniform(-3.0, 3.0, 2*m).reshape(2, m)
            a = sigmoid(np.dot(ws.T, xs) + b0)
            y = np.array([1 if a[0, i] > 0.5 else 0 for i in range(m)]).reshape(1, m)
        """
        test_cases = [
            {
                "name": "zero_check",
                "input": {
                    "x": np.array(
                        [
                             [2.56488323,  2.23455777,  0.50778764,  2.43102104, - 1.09900457, - 2.11195139, 2.15720461, - 0.3977942, - 0.03887522, - 0.28461782],
                             [-0.82107195, - 1.426292,    0.54797023, - 1.13993413, - 2.73777212,  1.24878165, - 1.88934367, - 1.85570881,  1.40969438, - 2.24432281],
                        ]
                    ),
                    "y": np.array(
                        [
                            [0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
                        ]
                    ),
                    "w": np.zeros((2, 1)),
                    "b": 0.0,
                    "learning_rate": 0.001,
                    "num_iters": 1000,
                },
                "expected": {
                    "cost": 0.4608720724511736,
                    "w": np.array([[-0.42980799], [0.18298383]]),
                    "b": 0.051742412959732836,
                },
            },

        ]
        for test_case in test_cases:
            result_cost, result_w, result_b, _ = gradientDescent(**test_case["input"])
            self.assertTrue(np.isclose(result_cost, test_case["expected"]["cost"]))
            self.assertTrue(np.isclose(result_w, test_case["expected"]["w"]).all())
            self.assertTrue(np.isclose(result_b, test_case["expected"]["b"]))


if __name__ == '__main__':
    unittest.main()
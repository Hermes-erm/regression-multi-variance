import numpy as np
import matplotlib.pyplot as plt


class MultipleRegression:
    def __init__(
        self,
        weights: float = [],  # [x1, x2, x3,... xn]
        bias: float = 0.0,
        learning_rate=0.01,
        batch=100,
        epochs=5,
    ):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.batch = batch
        self.epochs = epochs

        self.loss = []

    def get_parameters(self):
        return f"""Weights: {self.weights}
Bias: {self.bias}
Learning rate: {self.learning_rate}
Batch: {self.batch}
Epochs: {self.epochs}"""

    def predic(self, input):
        pass


def normalize(data: np.array):
    return (data - data.min()) / (data.max() - data.min())  # normalization
    # return (data - data.mean()) / (data.max() - data.min()) # mean normalization


def standardize(data: np.array):
    return (data - data.mean()) / data.std()

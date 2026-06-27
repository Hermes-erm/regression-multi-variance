import numpy as np
import matplotlib.pyplot as plt


class MultipleRegression:
    def __init__(
        self,
        weights: float = [0.0, 0.0],  # [x1, x2, x3,... xn]
        bias: float = 0.0,
        learning_rate=0.01,
        batch=100,
        epochs=5,
    ):
        self.weights = np.array(weights).reshape(-1, 1)  # [n rows, 1 col]
        self.bias = bias
        self.learning_rate = learning_rate
        self.batch = batch
        self.epochs = epochs

        self.loss = []

        self.fig, (self.error) = plt.subplots()

    def get_parameters(self):
        return f"""Weights: {self.weights}
Bias: {self.bias}
Learning rate: {self.learning_rate}
Batch: {self.batch}
Epochs: {self.epochs}"""

    def train(self, X, Y, cost_funtion=False):
        batch_x = np.array_split(X, len(X) / self.batch)
        batch_y = np.array_split(Y, len(Y) / self.batch)

        for epoch in range(0, self.epochs):
            for itr in range(len(batch_x)):
                x = batch_x[itr][:5]
                y = batch_y[itr][:5]
                n = len(x)

                # Forward pass
                # col level addition (n x n) + bias
                transpose_x = np.transpose(x)  # x.values for np
                y_pred = (self.weights * transpose_x).sum(axis=0) + self.bias

                # Calculate loss
                mse = self._calculate_loss(y, y_pred)

                x = np.transpose(x)
                dw = (-2 / n) * (x * (y - y_pred)).sum(axis=1)
                db = (-2 / n) * np.sum(y - y_pred)

                self.weights -= self.learning_rate * dw.reshape(-1, 1)  # np.array(dw)
                self.bias -= self.learning_rate * db

                self.loss.append(mse)

        if cost_funtion:
            self._cost_funtion(self.loss)
        else:
            self.fig.clear()

    def _calculate_loss(self, actual_var, predicted_var):
        return np.mean((actual_var - predicted_var) ** 2)

    def _cost_funtion(self, X):
        self.error.plot(X, label="Training Loss", color="blue")

        self.error.set_title("Loss per Iteration")
        self.error.set_xlabel("Iteration / Epoch")
        self.error.set_ylabel("Loss Value")

    def get_accuracy(self, X, Y):
        transpose_x = np.transpose(X)
        y_pred = (self.weights * transpose_x).sum(axis=0) + self.bias

        n = len(Y)

        num = np.minimum(Y, y_pred)
        denom = np.maximum(Y, y_pred)

        score = (num / denom) * 100
        score = 1 / n * np.sum(score)

        return f"Accuracy: {score:.2f} %"

    def test_split(self, data, proportion: float):
        if proportion < 0 or proportion > 1:
            raise ValueError("Proportion out of limit")

        training_size = round(len(data) * (1 - proportion))
        testing_size = round(len(data) * proportion)

        return data[:training_size], data[(-1 * testing_size) : -1]


def normalize(data: np.array):
    return (data - data.min()) / (data.max() - data.min())  # normalization
    # return (data - data.mean()) / (data.max() - data.min()) # mean normalization


def standardize(data: np.array):
    return (data - data.mean()) / data.std()

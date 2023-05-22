import math
import numpy as np

from utils import get_zeros_vec, get_dot


def get_log_loss(y_true: list[float], y_train: list[float]) -> float:
    """
        Рассчет функции потерь основываясь на правильных ответах
        и ответах, которые выдает логическая регрессия
    :param y_true:
    :param y_train:
    :return:
    """
    sum_loss = 0
    for i in range(len(y_true)):
        loss = -y_true[i] * math.log(y_train[i]) - (1 - y_true[i]) * math.log(1 - y_train[i])
        sum_loss += loss
    return sum_loss / len(y_true)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def get_gradient_descent(x: list[list[float]],
                         y: list[float],
                         n: int,
                         current_w: list[float],
                         current_b: float,
                         learning_rate: float) -> (list[float], float):
    dw: list[float] = get_zeros_vec(n)
    db: float = .0

    for i in range(len(x)):
        z = get_dot(x[i], current_w) + current_b
        a = sigmoid(z)

        # Считаем градиент
        for k in range(n):
            dw[k] += (a - y[i]) * x[i][k]
        db += (a - y[i])

    for k in range(n):
        dw[k] /= len(x)
    db /= len(x)

    updated_w: list = get_zeros_vec(n)

    for i in range(n):
        updated_w[i] = current_w[i] - learning_rate * dw[i]

    updated_b = current_b - learning_rate * db

    return updated_w, updated_b


class LogisticRegression:
    """
        Реализация логической регрессии
    """

    def __init__(self, n: int, x_test: list[list[float]], y_test: list[float]) -> None:
        self._n = n
        self._w = list(np.random.randn(n, 1) * 0.001)
        self._b = np.random.randn() * 0.001

        print(self._w, self._b)
        self._report_current_results = 40

        self._x_test = x_test
        self._y_test = y_test

        self._losses_train = []
        self._losses_test = []

    @property
    def report_current_results(self) -> int:
        return self._report_current_results

    @property
    def losses_train(self) -> list[float]:
        return self._losses_train

    @property
    def losses_test(self) -> list[float]:
        return self._losses_test

    def train(self, x: list[list[float]], y: list[float], learning_rate: float = 0.005, epochs=10) -> None:
        for epoch in range(epochs):
            gradient_values = get_gradient_descent(x, y, self._n, self._w, self._b, learning_rate)
            updated_w, updated_b = gradient_values

            self._w = updated_w
            self._b = updated_b

            if epoch % self._report_current_results == 0:
                self._losses_train.append(get_log_loss(y, self.predict(x)))
                self._losses_test.append(get_log_loss(self._y_test, self.predict(self._x_test)))

    def predict(self, x: list[list[float]]) -> list[float]:
        result: list[float] = []
        for i in range(len(x)):
            z = get_dot(x[i], self._w) + self._b
            s = sigmoid(z)
            result.append(s)
        return result

import math

from utils import get_zeros_vec, get_random_numbers, get_random_number


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
    dw: list = get_zeros_vec(n)
    db: float = .0

    for i in range(len(x)):
        z = [x[i][j] * current_w[j] + current_b for j in range(len(x[i]))]
        a = sigmoid(z[0])

        # Считаем градиент
        for k in range(n):
            dw[k] += (a - y[i]) * x[i][k]
        db += (a - y[i])

    for k in range(n):
        dw[k] /= len(x)
    db /= len(x)

    updated_w: list = get_zeros_vec(n)
    updated_b: list = get_zeros_vec(n)

    for i in range(n):
        updated_w[i] = current_w[i] - learning_rate * dw[i]
        updated_b[i] = current_b - learning_rate * db

    return updated_w, updated_b


class LogicalRegression:
    """
        Реализация логической регрессии
    """

    def __init__(self, n: int, x_test: list[list[float]], y_test: list[float]) -> None:
        self._n = n
        self._w = get_random_numbers(n)
        self._b = get_random_number()
        self._report_current_results = 100

        self._x_test = x_test
        self._y_test = y_test

        self._losses_train = []
        self._losses_test = []

    def train(self, x: list[list[float]], y: list[float], learning_rate: float, epochs=10) -> None:
        for epoch in range(epochs):
            gradient_values = get_gradient_descent(x, y, self._n, self._w, self._b, learning_rate)
            updated_w, updated_b = gradient_values

            self._w = updated_w
            self._b = updated_b

            if epoch % self._report_current_results == 0:
                self._losses_train.append(get_log_loss(y, self.__predict(x)))
                self._losses_test.append(get_log_loss(self._y_test, self.__predict(self._x_test)))

    def __predict(self, x: list[list[float]]) -> list[float]:
        result: list[float] = []
        for i in range(len(x)):
            z = [x[i][j] * self._w[j] + self._b for j in range(len(x[i]))]
            s = sigmoid(z[0])
            result.append(s)
        return result

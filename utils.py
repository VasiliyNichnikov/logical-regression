import random


def get_zeros_vec(n: int) -> list:
    """
    Возвращает нулевой вектор
    :param n:
    :return:
    """
    return [0 for i in range(n)]


def get_dot(x: list, y: list) -> float:
    dot = 0.0
    for i in range(len(x)):
        item_x = x[i]
        item_y = y[i]
        dot += (item_x * item_y)

    return dot


def get_random_number() -> float:
    return random.random()


def get_random_numbers(n: int) -> list[float]:
    return [get_random_number() for i in range(n)]

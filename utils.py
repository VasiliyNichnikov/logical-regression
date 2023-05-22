import random


def get_zeros_vec(n: int) -> list:
    """
    Возвращает нулевой вектор
    :param n:
    :return:
    """
    return [0 for i in range(n)]


def get_random_number() -> float:
    return random.random()


def get_random_numbers(n: int) -> list[float]:
    return [get_random_number() for i in range(n)]

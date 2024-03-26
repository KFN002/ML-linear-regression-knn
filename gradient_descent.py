import numpy as np


def gradient_descent(func: callable, start_point: float, gamma: float, epsilon: float, steps=0) -> np.array:
    history = []
    history.append(np.array(start_point))
    x = start_point

    if steps == 0:
        while True:
            grad = numerical_gradient(func, x)
            x_new = x - gamma * grad
            history.append(x_new)
            if abs(func(x_new) - func(x)) < epsilon:
                break
            x = x_new
    else:
        for _ in range(steps):
            grad = numerical_gradient(func, x)
            x_new = x - gamma * grad
            history.append(x_new)
            x = x_new

    return np.round(np.array(history), 3).reshape(-1, 1)


def numerical_gradient(func, x, delta=1e-9):
    return (func(x + delta) - func(x)) / delta


def test_func(x):
    return x ** 2


start_point = 1
gamma = 0.1
epsilon = 1e-5
max_steps = 0

history = gradient_descent(test_func, start_point, gamma, epsilon, steps=max_steps)

print(history)

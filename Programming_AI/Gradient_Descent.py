import random


def random_bias_theta(n_features: int) -> tuple[float, list[float]]:
    b = random.random()
    theta = [random.random() for _ in range(n_features)]
    return b, theta


def calculate_y(b: float, theta: list[float], x: list[float]) -> float:
    y_hat = 0.0
    for i in range(len(theta)):
        y_hat += theta[i] * x[i]
    return y_hat + b


def calculate_d_b(Y: list[float], Y_hat: list[float]) -> float:
    n = len(Y)
    total = 0.0
    for i in range(n):
        total += (Y[i] - Y_hat[i])
    return (-2 / n) * total


def calculate_d_theta(X: list[list[float]], Y: list[float], Y_hat: list[float]) -> list[float]:
    n = len(Y)
    n_features = len(X[0])
    grads = [0.0 for _ in range(n_features)]

    for j in range(n_features):
        total = 0.0
        for i in range(n):
            total += (Y[i] - Y_hat[i]) * X[i][j]
        grads[j] = (-2 / n) * total

    return grads


def update(
    X: list[list[float]],
    Y: list[float],
    Y_hat: list[float],
    b_prev: float,
    theta_prev: list[float],
    learning_rate: float
) -> tuple[float, list[float]]:
    d_theta = calculate_d_theta(X, Y, Y_hat)
    d_b = calculate_d_b(Y, Y_hat)

    b_new = b_prev - learning_rate * d_b
    theta_new = [
        theta_prev[j] - learning_rate * d_theta[j]
        for j in range(len(theta_prev))
    ]

    return b_new, theta_new


def fit(
    X: list[list[float]],
    Y: list[float],
    num_iterations: int,
    learning_rate: float = 0.2
) -> tuple[float, list[float]]:
    b, theta = random_bias_theta(len(X[0]))

    for _ in range(num_iterations):
        Y_hat = []
        for i in range(len(X)):
            Y_hat.append(calculate_y(b, theta, X[i]))

        b, theta = update(X, Y, Y_hat, b, theta, learning_rate)

    return b, theta


def solution(
    x_train: list[list[float]],
    y_train: list[float],
    x_test: list[list[float]],
    iterations: int = 1000
) -> list[float]:
    random.seed(42)
    b, theta = fit(x_train, y_train, iterations)
    return [round(calculate_y(b, theta, x), 2) for x in x_test]
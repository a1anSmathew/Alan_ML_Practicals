import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing


def data_load():
    X, y = fetch_california_housing(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y, columns=['target'])  # Naming target column
    return X, y


def Scaled(X):
    means = X.mean(axis=0)
    std_devs = X.std(axis=0)
    std_devs[std_devs == 0] = 1  # Prevent division by zero
    X_scaled = (X - means) / std_devs
    return X_scaled


def split_data(X_scaled, y):
    train = int(0.7 * X_scaled.shape[0])
    X_train = X_scaled.iloc[:train]
    X_test = X_scaled.iloc[train:]
    y_train = y.iloc[:train]
    y_test = y.iloc[train:]
    return X_train, X_test, y_train, y_test


def add_bias(X):
    X.insert(0, 'bias', 1)  # Add bias column
    return X


def initial_theta(X_train):
    d = X_train.shape[1]  # Number of features (including bias)
    return np.zeros(d)


def hypothesis_func(X_train, theta_values):
    return X_train.dot(theta_values)  # Vectorized hypothesis calculation


def Computing_error(y_train, y_pred):
    return y_pred - y_train.values.flatten()  # Compute residuals


def Computing_gradient(error_list, X_train):
    n = X_train.shape[0]
    return (X_train.T.dot(error_list)) / n  # Vectorized gradient computation


def updating_theta(gradient, theta_values, alpha=0.01):
    return theta_values - alpha * gradient  # Vectorized theta update


def cost_func(error_list):
    n = len(error_list)
    return np.sum(error_list ** 2) / (2 * n)  # Mean squared error


def main():
    X, y = data_load()
    X_scaled = Scaled(X)
    X_scaled = add_bias(X_scaled)  # Add bias after scaling
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    theta_values = initial_theta(X_train)

    for i in range(5000):  # Increased iterations for better convergence
        y_pred = hypothesis_func(X_train, theta_values)
        error_list = Computing_error(y_train, y_pred)
        gradient = Computing_gradient(error_list, X_train)
        theta_values = updating_theta(gradient, theta_values, alpha=0.01)  # Increased learning rate
        if i % 100 == 0:
            cost_function = cost_func(error_list)
            print(f"Iteration {i + 1}, Cost: {cost_function:.6f}")

    y_test_pred = hypothesis_func(X_test, theta_values)
    error_list = Computing_error(y_test, y_test_pred)
    final_cost = cost_func(error_list)
    print("Final Cost Function Value:", final_cost)

    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.show()

    r2 = r2_score(y_test, y_test_pred)
    print("R2 Score:", r2)


if __name__ == '__main__':
    main()

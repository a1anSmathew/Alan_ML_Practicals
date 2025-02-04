import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def load_data():
    df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X_vector = df[["age", "BMI", "BP", "blood_sugar", "Gender"]]

    y_vector = df["disease_score"]
    return X_vector, y_vector


def main():
    X_vector, y_vector=load_data()
    split_index = int(0.7 * len(X_vector))
    X_train, X_test = X_vector[:split_index], X_vector[split_index:]
    y_train, y_test = y_vector[:split_index], y_vector[split_index:]

    # X_scaled = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    # print(X_scaled)
    # y_scaled = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
    #
    # X_t_scaled = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    # y_t_scaled = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

    theta=np.dot((np.linalg.inv(np.dot(X_train.T,X_train))),np.dot(X_train.T,y_train))
    print("Theta: ", theta)

    y4=np.dot(X_test, theta)
    r2=r2_score(y_test, y4)
    print("R2 Score : ",r2)

    plt.scatter(y_test, y4)  # Provide x and y data directly
    plt.xlabel("True Values")  # Label for x-axis
    plt.ylabel("Predicted Values")  # Label for y-axis
    plt.title("True vs Predicted Values")  # Add a title
    plt.show()

    plt.scatter(X_test["age"],y4)
    plt.xlabel("Age")  # Label for x-axis
    plt.ylabel("Predicted Values")  # Label for y-axis
    plt.title("Age vs Predicted Values")  # Add a title
    plt.show()

    return y4


if __name__=="__main__":
    main()
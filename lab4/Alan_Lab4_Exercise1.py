import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def data_load():
    simulated_data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X=simulated_data.iloc[:,0:-2] #Last 2 columns are excluded from Training set
    y=simulated_data["disease_score"] #Ground Truth Values
    return (X,y)

# def Scaled(X,y):
#     means = X.mean(axis=0)
#     std_devs = X.std(axis=0)
#     X_scaled = (X - means) / std_devs
#     y_mean = y.mean()
#     y_std = y.std()
#     y_scaled = (y - y_mean) / y_std
#     return X_scaled,y_scaled


def split_data(X,y):
    train=int(0.7*(X.shape[0]))
    X_train=X.iloc[:train]
    X_test=X.iloc[train:]
    y_train=y.iloc[:train]
    y_test=y.iloc[train:]
    return (X_train,X_test,y_train,y_test)

def initial_theta(X_train):
    n=X_train.shape[0]
    d=X_train.shape[1]
    Theta_values=[]
    for i in range(d):
        theta = 0
        Theta_values.append(theta)
    return Theta_values

def hypothesis_func(X_train,theta_Values):
    n = X_train.shape[0]
    d = X_train.shape[1]
    y1=[]
    for i in range (n):
        total = 0
        for j in range (d):
            value = X_train.iloc[i, j] * theta_Values[j]
            total += value
        y1.append(total)
    return (y1)

def Computing_error(y_train,y1):
    n = y_train.shape[0]
    error_list = []
    for i in range (n):
        Error = ( y1[i] - y_train.iloc[i] )
        error_list.append(Error)
    return (error_list)

def Computing_gradient(error_list,X_train):
    n = X_train.shape[0]
    d = X_train.shape[1]
    gradient=[]
    for i in range(d):
        value=0
        for j in range(n):
            value += error_list[j] * X_train.iloc[j,i]
        gradient.append(value)
    return (gradient)

def updating_theta(gradient,theta_values):
    alpha = 0.0000001
    for i in range (len(theta_values)):
        theta_values[i] = theta_values[i] - alpha * gradient[i]
    return (theta_values)

def cost_func(error_list):
    total_error=sum(error**2 for error in error_list)
    cost_function = total_error/2
    return (cost_function)


def main():
    X,y = data_load()
    # X_scaled , y_scaled = Scaled(X,y)
    X_train , X_test , y_train , y_test = split_data(X,y)
    theta_values = initial_theta(X_train)
    for i in range (1000):
        y1 = hypothesis_func(X_train,theta_values)
        error_list = Computing_error(y_train,y1)
        gradient = Computing_gradient(error_list,X_train)
        theta_values = updating_theta(gradient, theta_values)
        cost_function=cost_func(error_list)


    y2 = hypothesis_func(X_test,theta_values)
    error_list = Computing_error(y_test,y2)
    cost_function=cost_func(error_list)
    print("Cost function of test data: ",cost_function)
    print(theta_values)

    sns.set_style("whitegrid")

    # Create figure and scatter plot
    plt.figure(figsize=(9, 5))
    plt.scatter(X_test["age"], y2, color="royalblue", alpha=0.7, edgecolors="black", linewidth=0.8)

    # Add labels and title
    plt.xlabel("Age", fontsize=14, fontweight="bold", color="darkblue")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold", color="darkblue")
    plt.title("Scatter Plot of Predicted Values vs Age", fontsize=16, fontweight="bold", color="darkred")

    # Add grid with customized style
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    sns.set_style("whitegrid")

    # Create figure and scatter plot
    plt.figure(figsize=(9, 5))
    plt.scatter(y_test, y2, color="royalblue", alpha=0.7, edgecolors="black", linewidth=0.8)

    # Add labels and title
    plt.xlabel("Given Y Values", fontsize=14, fontweight="bold", color="darkblue")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold", color="darkblue")
    plt.title("Scatter Plot of Predicted Values vs Age", fontsize=16, fontweight="bold", color="darkred")

    # Add grid with customized style
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    r2=r2_score(y_test,y2)
    print("R2 Score: ",r2)
    return y2






if __name__ == '__main__':
    main()
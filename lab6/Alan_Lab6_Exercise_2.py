import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def load_data():
    breast_cancer = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = breast_cancer.iloc[:,:-2]
    y = breast_cancer["disease_score"]
    return X,y


def split_data(X,y,a,b):
    X_test = X.iloc[a:b, :]
    X_train = X.drop(X.index[a:b])
    y_test = y.iloc[a:b]
    y_train = y.drop(y.index[a:b])
    return X_train,X_test,y_train,y_test

def Normalization(X_train):
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    X_scaled = (X_train - mins) / (maxs - mins)
    return X_scaled,mins,maxs

def Normalization2(X_test,mins,maxs):
    X_test_scaled = (X_test - mins) / (maxs - mins)
    return X_test_scaled

def initial_theta(X_scaled):
    d=X_scaled.shape[1]
    Theta_values=[0 for _ in range (d)]
    return Theta_values

def hypothesis_function(X_train,Theta_values):
    # n = X_train.shape[0]
    # d = X_train.shape[1]
    # y1_values = [sum(X_train.iloc[i,j] * Theta_values[j] for j in range (d))for i in range (n) ]

    Theta_values = np.array(Theta_values)
    y1_values = np.dot(X_train, Theta_values)  # X_train is (m x d), Theta_values is (d,)
    return y1_values

def error_list(y1_values,y_train):
    err_list = [y1_values[i] - y_train.iloc[i] for i in range(len(y_train))]
    RES = [(y1_values[i] - y_train.iloc[i]) ** 2 for i in range(len(y_train))]
    y_mean = y_train.mean()  # Using actual y_train mean
    TOT = [(y_train.iloc[i] - y_mean) ** 2 for i in range(len(y_train))]  # Total sum of squares based on actual values
    return err_list, sum(RES), sum(TOT)


def gradient(X_train,err_list):
    n = X_train.shape[0]
    d = X_train.shape[1]
    grad_list = [sum(err_list[i] * X_train.iloc[i,j] for i in range (n)) for j in range (d)]
    return grad_list

def updating_theta(grad_list,Theta_values):
    alpha = 0.00000000001
    for i in range (len(Theta_values)):
        Theta_values[i] = Theta_values[i] - alpha * grad_list[i]

    return (Theta_values)

def cost_function(err_list):
    total_error = sum(error ** 2 for error in err_list)
    cost_funct = total_error / 2
    return (cost_funct)


def main():
    X, y = load_data()
    m = X.shape[0]
    fold_size = int(0.10 * (m))
    a = 0
    b = fold_size
    for j in range (10):
        X_train, X_test, y_train, y_test = split_data(X,y,a,b)
        a = b
        b += fold_size
        if b > m:  # Ensure `b` doesn't exceed dataset size
            b = m
        X_scaled,mins,maxs = Normalization(X_train)
        Theta_values = initial_theta(X)
        for i in range (1000):
            y1_values = hypothesis_function(X_scaled,Theta_values)
            err_list,RES,TOT = error_list(y1_values, y_train)
            grad_list = gradient(X_train, err_list)
            Theta_values = updating_theta(grad_list, Theta_values)
            cost_func = cost_function(err_list)
            # print(cost_func)
            # print(sum(err_list))
        # X_scaled, mins, maxs = Normalization(X_train)
        X_test_scaled = Normalization2(X_test,mins,maxs)
        y1_values = hypothesis_function(X_test_scaled, Theta_values)
        err_list, RES , TOT = error_list(y1_values,y_test)
        print("Sum of Error of Training Set: " , sum(err_list))
        print("RES",RES)
        print("TOT",TOT)
        cost_func = cost_function(err_list)
        print("Cost Function", cost_func)

        r2 = 1 - (RES/TOT)
        print("R2 Score", r2)



if __name__ == '__main__':
    main()
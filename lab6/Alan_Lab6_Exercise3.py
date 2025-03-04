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

def scaled_data(X_train):
    # Separate gender column
    if 'gender' in X_train.columns:
        gender_col = X_train['gender']
        X_train = X_train.drop(columns=['gender'])
    else:
        gender_col = None

    means = X_train.mean(axis=0)
    std_devs = X_train.std(axis=0)
    std_devs[std_devs == 0] = 1  # Avoid division by zero

    X_scaled = (X_train - means) / std_devs

    # Add gender column back
    if gender_col is not None:
        X_scaled['gender'] = gender_col

    return X_scaled, means, std_devs

def scaled_data2(X_test, means, std_devs):
    if 'gender' in X_test.columns:
        gender_col = X_test['gender']
        X_test = X_test.drop(columns=['gender'])
    else:
        gender_col = None

    X_scaled = (X_test - means) / std_devs

    if gender_col is not None:
        X_scaled['gender'] = gender_col

    return X_scaled

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
    alpha = 0.000001
    for i in range (len(Theta_values)):
        Theta_values[i] = Theta_values[i] - alpha * grad_list[i]

    return (Theta_values)

def cost_function(err_list):
    total_error = sum(error ** 2 for error in err_list)
    cost_funct = total_error / 2
    return (cost_funct)


def main():
    X, y = load_data()
    # X_scaled = scaled_data(X)
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
        Theta_values = initial_theta(X)
        X_scaled, means, std_devs = scaled_data(X_train)
        for i in range (1000):
            y1_values = hypothesis_function(X_train,Theta_values)
            err_list,RES,TOT = error_list(y1_values, y_train)
            grad_list = gradient(X_train, err_list)
            Theta_values = updating_theta(grad_list, Theta_values)
            cost_func = cost_function(err_list)
            # print(cost_func)
            # print(sum(err_list))

        # X_test_scaled = scaled_data2(X_test,means, std_devs)
        y1_values = hypothesis_function(X_test, Theta_values)
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
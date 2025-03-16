import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_load():
    simulated_data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X=simulated_data.iloc[:,0:-2] #Last 2 columns are excluded from Training set
    y=simulated_data["disease_score"] #Ground Truth Values
    return (X,y)

def load_data():
    df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X_vector = df[["age", "BMI", "BP", "blood_sugar", "Gender"]]

    y_vector = df["disease_score"]
    return X_vector, y_vector

def Scaled(X):
    means = X.mean(axis=0)
    std_devs = X.std(axis=0)
    X_scaled = (X - means) / std_devs
    return X_scaled


def split_data(X_scaled,y):
    train=int(0.7*(X_scaled.shape[0]))
    X_train=X_scaled.iloc[:train]
    X_test=X_scaled.iloc[train:]
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
    print(Theta_values)
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
    alpha = 0.000001
    for i in range (len(theta_values)):
        theta_values[i] = theta_values[i] - alpha * gradient[i]
    return (theta_values)

def cost_func(error_list):
    total_error=sum(error**2 for error in error_list)
    cost_function = total_error/2
    return (cost_function)


def main():
    X,y = data_load()
    # X_scaled = Scaled(X)
    X_train , X_test , y_train , y_test = split_data(X,y)
    theta_values = initial_theta(X_train)
    for i in range (100):
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



    plt.scatter(y_test, y2)  # Provide x and y data directly
    plt.xlabel("True Values (Scaled)")  # Label for x-axis
    plt.ylabel("Predicted Values")  # Label for y-axis
    plt.title("True vs Predicted Values")  # Add a title
    plt.show()

    # plt.scatter(X_test["age"],y2)
    # plt.show()

    r2=r2_score(y_test,y2)
    print("R2 Score(from scratch): ",r2)
    # return y2

    X_vector, y_vector = load_data()
    split_index = int(0.7 * len(X_vector))
    X_train, X_test = X_vector[:split_index], X_vector[split_index:]
    y_train, y_test = y_vector[:split_index], y_vector[split_index:]

    # X_scaled = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    # y_scaled = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
    #
    # X_t_scaled = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    # y_t_scaled = (y_test - y_test.mean(axis=0)) / y_test.std(axis=0)

    theta = np.dot((np.linalg.inv(np.dot(X_train.T, X_train))), np.dot(X_train.T, y_train))
    print("Theta: ", theta)

    y4 = np.dot(X_test, theta)
    r2 = r2_score(y_test, y4)
    print("R2 Score for Closed Form",r2)

    plt.scatter(y_test, y4)  # Provide x and y data directly
    plt.xlabel("True Values (Scaled)")  # Label for x-axis
    plt.ylabel("Predicted Values y4")  # Label for y-axis
    plt.title("True vs Predicted Values")  # Add a title
    plt.show()

    # plt.scatter(X_test["age"], y4)
    # plt.show()

    # return y4

    X, y = data_load()
    # X_scaled = Scaled(X)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Training a linear regression
    model: LinearRegression = LinearRegression()

    # Training the mode
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute the r2 score
    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f (closer to 1 is good) " % r2)

    plt.scatter(y_test, y_pred)  # Provide x and y data directly
    plt.xlabel("True Values (Scaled)")  # Label for x-axis
    plt.ylabel("Predicted Values y_pred")  # Label for y-axis
    plt.title("True vs Predicted Values")  # Add a title
    plt.show()

    X_train, X_test, y_train, y_test = split_data(X, y)
    # X_test = X_test.iloc[:,0]
    X,y = data_load()
    train = int(0.7 * (X.shape[0]))
    X_age = X.iloc[train:,0]
    plt.figure(figsize=(10, 6))

    age_dict = {}
    for i in range (X_age.shape[0]):
        age_dict[X_age.iloc[i]] = [y_pred[i], y2[i], y4[i]]
    print(age_dict)


    # Plot each prediction on the same graph
    plt.scatter(X_age,y_test,label="Actual Data")
    plt.plot(X_age, y2, label="Gradient Descent", linestyle='-', color='blue', marker='o')
    plt.plot(X_age, y4, label="Normal Equation", linestyle='--', color='green')
    plt.plot(X_age, y_pred, label="Scikit", color='orange')

    # Add labels, title, legend, and grid
    plt.xlabel('Feature age')
    plt.ylabel('Predicted Values')
    plt.title('Comparison of Linear Regression Predictions')
    plt.legend(loc='best')  # Automatically chooses the best position
    plt.grid(alpha=0.5)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
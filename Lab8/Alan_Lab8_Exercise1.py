import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#l1 and l2 regularization from scratch.

alpha = 0.01  # Learning rate
iters = 1000  # Number of iterations
lmbd = 0.01   # Lambda value assigned

# Load dataset
def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    x1 = data.drop(columns=["disease_score", "disease_score_fluct"])
    y1 = data["disease_score_fluct"]
    x1.to_numpy()
    y1.to_numpy()
    x1_mean = np.mean(x1, axis=0)
    x1_sd = np.std(x1, axis=0)
    x1 = (x1 - x1_mean) / x1_sd
    # [X, y] = fetch_california_housing(return_X_y=True)
    # split the data: train 70%, test 30%
    X_train, X_test, y_train,y_test = train_test_split(x1, y1, test_size = 0.30, random_state = 999)  #simulated dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999) #california housing dataset
    return X_train, y_train, X_test,y_test

x_train, y_train, x_test, y_test = load_data()

# Prepare data
def prepare_data(x_train, y_train,x_test,y_test):
    n1 = x_train.shape[0]
    x_train = np.c_[np.ones(n1), x_train]
    y_train = y_train.values.reshape(-1,1)
    n2 = x_test.shape[0]
    x_test = np.c_[np.ones(n2), x_test]
    y_test = y_test.values.reshape(-1, 1)
    return x_train, y_train, x_test, y_test

x1,y1,x2,y2 = prepare_data(x_train, y_train,x_test,y_test)

# Initialize theta
def theta_first(x1):
    theta_one = np.zeros(x1.shape[1], dtype=float)
    return theta_one

#computing the theta values using normal values to compare with my implementation of the GDA:
def norm_eq(x1, y1):
    X_transpose = x1.T
    theta = np.linalg.inv(X_transpose @ x1) @ X_transpose @ y1
    return theta


# Compute cost function
def compute_cost(x1, y1, theta):
    h_theta = np.dot(x1, theta)
    cost = (1 / 2) * np.sum((h_theta - y1) ** 2)
    return cost

def l2_reg(theta):
    l2_norm = np.dot(theta.T,theta)
    penalty = lmbd * l2_norm
    return penalty

def l1_reg(theta):
    l1_norm = np.sum(np.abs(theta))
    penalty = lmbd * l1_norm
    return penalty

#gradient descent
def gradient_descent(x1,y1):
    m = len(y1)
    theta = theta_first(x1).reshape(-1, 1)
    cost_vals = []

    for _ in range(iters):
        h_theta = np.dot(x1, theta)
        gradient = (1 / m) * np.dot(x1.T, (h_theta - y1))
        theta -= alpha * gradient  # Update theta
        cost = compute_cost(x1, y1, theta)
        cost += l2_reg(theta)          #ridge    R2:  0.56
        # cost += l1_reg(theta)            #lasso    R2: 0.56
        cost_vals.append(cost)

    return theta, cost_vals

theta_mod, costs = gradient_descent(x1, y1)

print("Optimized Theta:")
print(theta_mod)
print("Cost History:")
print(costs)

# #plotting the data:
# plt.plot(range(iters), costs, color='blue')
# plt.title('Cost Function over Iterations')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Cost')
# plt.show()

#cost calc using normal equation
theta_norm = norm_eq(x1, y1)
y_pred_norm = np.dot(x2,theta_norm)
print("The theta values using the normal equation are:")
print(norm_eq(x1, y1))

#now testing:
y_pred = np.dot(x2, theta_mod)  #sigma h(theta)x.
denominator = np.sum((y2 - np.mean(y2)) ** 2)
numerator = np.sum((y2 - y_pred) ** 2)
r2 = 1 - (numerator / denominator)

print("The R^2 score for this model on this dataset is:",r2)
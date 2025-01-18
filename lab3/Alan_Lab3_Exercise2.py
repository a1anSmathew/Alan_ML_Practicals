from random import random
import numpy as np
import pandas as pd
from random import randint


def Exercise2():
    simulated_data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X=simulated_data.iloc[:,0:-2]
    y=simulated_data["disease_score"]

#Splitting the data
    rows=int(0.7*X.shape[0])
    X_train=X.iloc[:rows, :]
    X_test=X.iloc[rows:, :]
    y_train=y.iloc[:rows]
    y_test=y.iloc[rows:]


#Initializing the theta values to zero
    n=X_train.shape[0]
    d=X_train.shape[1]
    Theta_values=[]
    for i in range(d):
        theta = 0
        Theta_values.append(theta)
    print(Theta_values)


#Computing the hypothesis function(y1) and Sum of Squared Error and Cost function
    k = 0
    E=0
    Error_List=[]
    y1_values=[]
    while k <= n-1:
        y1 = 0
        for i in range(d):
            theta = Theta_values[i]
            X_val = X_train.iloc[k,i]
            sum = theta * X_val
            y1 += sum
        y1_values.append(y1)
        y_val = y.iloc[k]
        Err = (y1 - y_val) ** 2
        E += (y1 - y_val) ** 2
        Error_List.append(Err)
        k += 1
    print("Hypothesis function: ",y1_values)
    print(len(y1_values))
    print("Sum of Square Errors :",E)
    print("Cost Function: ",E/(2*n))
    print("Error List: ",Error_List)

#Finding the Derivative / Gradient
    y1_y=[]
    for i in range (n-1):
        val=y1_values[i] - y_train[i]
        y1_y.append(val)
    # print(len(y1_y))

    deri_list = []
    # for i in range(len(y1_y)):
    i=0
    for j in range(d-1):
        sum=0
        for k in range (n-1):
            val = y1_y[i] * X_train.iloc[k, j]
            sum += val
        deri_list.append(sum)
        i += 1
    print(deri_list)

# #Updating the theta values
#     alpha=0.1
#     for Theta in Theta_values :
#         Theta = Theta - alpha *



                                            
def main():
    Exercise2()

if __name__ == '__main__':
    main()
import time
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR

import matplotlib.pyplot as plt


def load_data():
    [X, y] = fetch_california_housing(return_X_y=True, as_frame=True)

    #Split data - train=70% and test=30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999) #Test size is set to 30%

    #Scale the data(Standardizing the data)
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Training a linear regression
    model: LinearRegression=LinearRegression()

    #Training the mode
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    #Compute the r2 score
    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f (closer to 1 is good) " % r2)

def main():
    load_data()

if __name__ == '__main__':
    main()
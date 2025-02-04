import time
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVR

import matplotlib.pyplot as plt


def load_data():
    breast_cancer = pd.read_csv("data.csv")
    X = breast_cancer.iloc[:,2:-1]
    le = LabelEncoder()
    y = breast_cancer["diagnosis"]
    y = le.fit_transform(y)
    # y = y.replace({"M":1,"B":0})

    #Split data - train=70% and test=30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999) #Test size is set to 30%

    #Scale the data(Standardizing the data)
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Training a linear regression
    model: LogisticRegression = LogisticRegression()

    #Training the mode
    model.fit(X_train_scaled,y_train)

    y_pred=model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

def main():
    load_data()

if __name__ == '__main__':
    main()
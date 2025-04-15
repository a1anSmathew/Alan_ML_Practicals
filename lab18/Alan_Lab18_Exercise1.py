import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def load_data(x1,x2,labels):
    data = []
    data = pd.DataFrame(data)
    data["x1"] = x1
    data["x2"] = x2
    data["y"] = labels
    X = data[["x1","x2"]]
    y = data["y"]
    le=LabelEncoder()
    y = le.fit_transform(y)
    return X, y

def RBF_kernel(X,y):
    X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.7,random_state=44)
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='rbf',gamma=0.1,C=10)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)

    print("Accuracy Score of RBF Kernel is: ",acc)

def Polynomial_kernel(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=44)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='poly', degree=3, gamma='scale', C=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy Score of Polynomial Kernel is: ", acc)

def main():
    x1 = [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14]
    x2 = [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8]
    labels = ['Blue', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Red', 'Red', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']

    X,y = load_data(x1,x2,labels)

    print("|--------------RBF KERNEl------------|")
    RBF_kernel(X,y)

    print("|--------------POLYNOMIAL KERNEl------------|")
    Polynomial_kernel(X,y)

if __name__ == '__main__':
    main()
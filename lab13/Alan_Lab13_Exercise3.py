import pandas as pd
from sklearn import datasets
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def load_data():
    diabetes = datasets.load_diabetes()
    X1 = diabetes.data
    y1 = diabetes.target
    df = pd.DataFrame(X1, columns=diabetes.feature_names)
    df['target'] = y1

    return X1,y1

def load_data2():
    diabetes = datasets.load_iris()
    X2 = diabetes.data
    y2 = diabetes.target
    # df = pd.DataFrame(X2, columns=diabetes.feature_names)
    # df['target'] = y2

    return X2,y2

def Random_forest_reg(X1,y1):
    X_train , X_test , y_train , y_test = train_test_split(X1,y1,test_size=0.3 , random_state=44)
    model = RandomForestRegressor(n_estimators=100 , random_state=44,max_depth=None)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    print("R2 Score is ",r2)

def Random_forest_class(X2,y2):
    X_train , X_test , y_train , y_test = train_test_split(X2,y2,test_size=0.3,random_state=42)
    model1 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None)
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score is ", acc)


def main():
    X1,y1 = load_data()
    X2,y2 = load_data2()
    Random_forest_reg(X1,y1)
    Random_forest_class(X2,y2)

if __name__ == '__main__':
    main()
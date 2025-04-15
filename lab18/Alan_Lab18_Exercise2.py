from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data():
    data = load_iris()
    X = data.data
    X = X[:,0:2]
    y = data.target

    return X,y

def kernel_methods(X,y):
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.9,random_state=44,stratify=y)
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    for kernel in kernels:
        model = SVC(kernel=kernel,gamma=0.1,C=10)
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test,y_pred)

        print(f"Accuracy Score of {kernel} Kernel is: ",acc)

def main():
    X,y = load_data()
    kernel_methods(X,y)

if __name__ == '__main__':
    main()
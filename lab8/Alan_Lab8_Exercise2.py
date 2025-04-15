import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")
X = data.drop(columns=["id", "Unnamed: 32","diagnosis"])
y = data["diagnosis"]
y = y.map({'M': 1, 'B': 0})

# sonar = pd.read_csv("sonar.csv",header=None)                       #tried with multiple datasets
# X = sonar.iloc[:, :-1]  # All columns except the last one
# y = sonar.iloc[:, -1]   # Last column (target)
# y = y.map({'M': 1, 'R': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lambda_values = [0.01, 0.1, 1, 10, 100]  #larger lambda means more regularization
C_values = [1 / lmbda for lmbda in lambda_values]
ridge = []
lasso = []
ridge_thetas=[]
lasso_thetas=[]

for each in C_values:
    ridge_clf = LogisticRegression(penalty='l2', solver='liblinear', C=each)  #C is inverse of lambda
    ridge_clf.fit(X_train, y_train)
    ridge_preds = ridge_clf.predict(X_test)
    ridge.append(ridge_preds)
    ridge_thetas.append(ridge_clf.coef_)  # Store coefficients

    #lasso
    lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', C=each)
    lasso_clf.fit(X_train, y_train)
    lasso_preds = lasso_clf.predict(X_test)
    lasso.append(lasso_preds)
    lasso_thetas.append(lasso_clf.coef_)  # Store coefficients

#evaluate models
list_ridge_acc=[]
list_lasso_acc=[]
for x,y in zip(ridge,lasso):
    ridge_acc = accuracy_score(y_test, x)
    lasso_acc = accuracy_score(y_test, y)
    list_ridge_acc.append(ridge_acc)
    list_lasso_acc.append(lasso_acc)

max_ridge_index = list_ridge_acc.index(max(list_ridge_acc))
max_lasso_index = list_lasso_acc.index(max(list_lasso_acc))

print("Maximum Ridge Classifier Accuracy:", max(list_ridge_acc), "for lambda value:",lambda_values[max_ridge_index])
print("The corresponding theta values for ridge:")
print(lasso_thetas[max_lasso_index])
print("Maximum Lasso Classifier Accuracy:", max(list_lasso_acc), "for lambda value:",lambda_values[max_lasso_index])
print("The corresponding theta values for lasso:")
print(lasso_thetas[max_ridge_index])


#without regularization:
def sci_train():
    # Logistic regression using scikit
    # sonar = pd.read_csv("sonar.csv", header=None)  # tried with multiple datasets
    # X = sonar.iloc[:, :-1]  # All columns except the last one
    # y = sonar.iloc[:, -1]  # Last column (target)
    # y = y.map({'M': 1, 'R': 0})

    data = pd.read_csv("data.csv")
    X = data.drop(columns=["id", "Unnamed: 32", "diagnosis"])
    y = data["diagnosis"]
    y = y.map({'M': 1, 'B': 0})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=999)

    model = LogisticRegression(max_iter=1000, random_state=999, penalty='l1', solver='liblinear', C=1.0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, model.coef_

accuracy, coefficients = sci_train()
print("Accuracy without regularization:", accuracy)
print("Theta values without optimization:",coefficients)





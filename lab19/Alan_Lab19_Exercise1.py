import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def Logistic_reg():
    iris = pd.read_csv('heart.csv')
    X = iris.iloc[:,:-1]
    y = iris.iloc[:,-1]

    X_train , X_test , y_train , y_test = train_test_split(X,y,train_size=0.7,random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=20000)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_pred,y_test)
    print("Accuracy of the model is: ",acc)

    y_probabilities = model.predict_proba(X_test)
    y_proba = y_probabilities[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Sensitivity (Recall): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return y_pred,y_test

def Evaluation(y_pred,y_test):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i,j in zip(y_test,y_pred):
        if i == 1 and j == 1:
            TP += 1
        elif i == 1 and j == 0:
            FN += 1
        elif i == 0 and j == 1:
            FP +=1
        else:
            TN += 1

    print(TP,FP,TN,FN)



def main():
    y_pred , y_test = Logistic_reg()
    Evaluation(y_pred,y_test)


if __name__ == '__main__':
    main()


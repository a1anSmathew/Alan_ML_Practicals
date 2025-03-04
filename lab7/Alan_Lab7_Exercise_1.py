import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_data():
    data = pd.read_csv("sonar.csv")
    X = data.iloc[:, :-1]  # Features (all columns except the last)
    y = data.iloc[:, -1]  # Target (last column)
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
    return X_train, X_test, y_train, y_test


def scaling(X_train, X_test, y_train, y_test):
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode labels (y) using LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test


# Define k-Fold cross-validation (e.g., k=5)
def k_fold(X_train, y_train):
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Logistic Regression Model
    model = LogisticRegression(max_iter=200)

    # Perform cross-validation and print scores
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    print(f'Cross-validation accuracy scores: {scores}')
    print(f'Average accuracy: {scores.mean():.4f}')


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, y_train, y_test = scaling(X_train, X_test, y_train, y_test)
    k_fold(X_train, y_train)


if __name__ == '__main__':
    main()

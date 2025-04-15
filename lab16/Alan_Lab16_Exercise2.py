import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def XGBoost_Classifier():
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost Classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',  # Binary classification
        n_estimators=100,            # Number of boosting rounds
        max_depth=3,                 # Tree depth
        learning_rate=0.1,           # Step size shrinkage
        subsample=0.8,               # Fraction of samples used per tree
        colsample_bytree=0.8,        # Fraction of features used per tree
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def XGBoost_Regressor():
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Regression task
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("RMSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

def main():
    XGBoost_Classifier()
    XGBoost_Regressor()

if __name__ == '__main__':
    main()
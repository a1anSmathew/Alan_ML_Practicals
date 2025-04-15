import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

def gradient_boosting_regression():
    boston = pd.read_csv('Boston.csv')
    X_boston = boston.drop(columns=['medv'])  # Target variable is 'medv'
    y_boston = boston['medv']

    X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)

    gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_regressor.fit(X_train, y_train)

    y_pred = gb_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Gradient Boosting Regression MSE: {mse:.2f}")

def gradient_boosting_classification():
    weekly = pd.read_csv('Weekly.csv')
    X_weekly = weekly.drop(columns=['Direction'])  # Target variable is 'Direction'
    y_weekly = weekly['Direction'].apply(lambda x: 1 if x == 'Up' else 0)  # Convert to binary

    X_train, X_test, y_train, y_test = train_test_split(X_weekly, y_weekly, test_size=0.2, random_state=42)

    gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_classifier.fit(X_train, y_train)

    y_pred = gb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Classification Accuracy: {accuracy:.2f}")

gradient_boosting_regression()
gradient_boosting_classification()
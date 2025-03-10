import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

# Step 1: Load Data and Convert `y` to Numeric
def load_data():
    data = pd.read_csv("data.csv")
    X = data.iloc[:, 2:]  # Features

    # Convert `y` (categorical) to numeric using Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data.iloc[:, 1])  # Encoding target variable

    return X, y, label_encoder  # Return encoder for possible decoding

# Step 2: Train Regression Decision Tree
def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Regressor
    reg_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg_tree.fit(X_train, y_train)

    # Make predictions
    y_pred = reg_tree.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return reg_tree, X_test, y_test, y_pred

# Step 3: Visualizing the Decision Tree
def visualize_tree(reg_tree, feature_names):
    plt.figure(figsize=(15, 8))
    tree.plot_tree(reg_tree, feature_names=feature_names, filled=True, rounded=True, fontsize=8)
    plt.show()

# Main function to execute the workflow
def main():
    X, y, label_encoder = load_data()  # Load data and encoder
    reg_tree, X_test, y_test, y_pred = train_decision_tree(X, y)

    # Decode predictions (optional, if needed)
    decoded_y_pred = label_encoder.inverse_transform(np.round(y_pred).astype(int))
    print(f"Decoded Predictions: {decoded_y_pred[:10]}")  # Show first 10 predictions

    # Visualizing the Decision Tree
    visualize_tree(reg_tree, X.columns)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
def load_data():
    simulated_data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = simulated_data.iloc[:, 0:-2].values  # Last 2 columns are excluded from Training set
    y = simulated_data["disease_score"].values  # Ground Truth Values
    return (X, y)


# Node class for decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Function to split data
def partition(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


# Calculate Mean Squared Error
def mse(y):
    if len(y) == 0:
        return 0
    return np.var(y) * len(y)


# Find the best feature and threshold
def best_split(X, y):
    best_mse = float("inf")
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        sorted_indices = np.argsort(X[:, feature_index])
        for j in range(len(y) - 1):
            threshold = (X[sorted_indices[j], feature_index] + X[sorted_indices[j + 1], feature_index]) / 2
            X_left, X_right, y_left, y_right = partition(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            total_mse = mse(y_left) + mse(y_right)
            if total_mse < best_mse:
                best_mse, best_feature, best_threshold = total_mse, feature_index, threshold

    return best_feature, best_threshold


# Build Decision Tree Regressor
def build_tree(X, y, depth=0, max_depth=5):
    if len(y) == 0:
        return None

    if depth >= max_depth or len(set(y)) == 1:
        return Node(value=np.mean(y))

    feature, threshold = best_split(X, y)

    if feature is None:
        return Node(value=np.mean(y))

    X_left, X_right, y_left, y_right = partition(X, y, feature, threshold)

    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth)

    return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)


# Predict single sample
def predict(node, X):
    if node.value is not None:
        return node.value
    if X[node.feature] <= node.threshold:
        return predict(node.left, X)
    else:
        return predict(node.right, X)


# Predict multiple samples
def predict_batch(tree, X):
    return np.array([predict(tree, sample) for sample in X])


# Print tree structure
def print_tree(node, depth=0):
    if node.value is not None:
        print(f"{'  ' * depth}Leaf: Value {node.value:.2f}")
    else:
        print(f"{'  ' * depth}Feature {node.feature} <= {node.threshold:.2f}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)


# Main execution
if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

    # Train Decision Tree Regressor
    tree = build_tree(X_train, y_train, max_depth=5)

    # Make predictions
    predictions = predict_batch(tree, X_test)
    mse_score = mean_squared_error(y_test, predictions)
    R2 = r2_score(y_test, predictions)
    print(R2)

    print("Decision Tree Regression MSE:", mse_score)
    print_tree(tree)

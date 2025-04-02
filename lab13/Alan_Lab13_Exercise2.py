import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets


# Load dataset
def load_data():
    # simulated_data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    # X = simulated_data.iloc[:, 0:-2].values  # Last 2 columns are excluded from Training set
    # y = simulated_data["disease_score"].values  # Ground Truth Values
    # return (X, y)

    simulated_data = datasets.load_diabetes()
    X = simulated_data.data
    y = simulated_data.target

    return X,y

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


# Bagging Regressor
def bagging_regressor(X_train, y_train, num_trees=10, max_depth=5):
    trees = []
    n_samples = len(y_train)

    for i in range(num_trees):
        indices = np.random.choice(n_samples, n_samples, replace=True)  # Bootstrap sampling (number_of_samples, number_of_draws, replace=True/False)
        X_sample, y_sample = X_train[indices], y_train[indices] # Select only the bootstrapped indices
        tree = build_tree(X_sample, y_sample, max_depth=max_depth) # Build tree for the bootstrapped indices
        trees.append(tree)
        predictions = predict_batch(tree, X_test)
        mse_score = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Tree {i + 1}: MSE = {mse_score:.4f}, R² = {r2:.4f}")

    return trees


# Predict using Bagging Regressor
def predict_bagging(trees, X):
    predictions = np.array([predict_batch(tree, X) for tree in trees])
    return np.mean(predictions, axis=0)


# Main execution
if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

    # Train Bagging Regressor
    trees = bagging_regressor(X_train, y_train, num_trees=10, max_depth=5)

    # Make predictions
    predictions = predict_bagging(trees, X_test)
    mse_score = mean_squared_error(y_test, predictions)
    R2 = r2_score(y_test, predictions)
    print("Bagging Regressor R2:", R2)
    print("Bagging Regressor MSE:", mse_score)

    # Print one of the decision trees
    print_tree(trees[0])

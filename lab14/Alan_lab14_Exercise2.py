import numpy as np
from sklearn import datasets


# Load dataset
def load_data():
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for AdaBoost
    return X, y


# Node class for decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Partition function
def partition(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


# Calculate Gini impurity
def gini(y):
    if len(y) == 0:
        return 0
    p = np.sum(y == 1) / len(y)
    return 1 - p ** 2 - (1 - p) ** 2


# Find best split for classification
def best_split(X, y):
    best_gini = float("inf")
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        sorted_indices = np.argsort(X[:, feature_index])
        for j in range(len(y) - 1):
            threshold = (X[sorted_indices[j], feature_index] + X[sorted_indices[j + 1], feature_index]) / 2
            X_left, X_right, y_left, y_right = partition(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            total_gini = (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / len(y)
            if total_gini < best_gini:
                best_gini, best_feature, best_threshold = total_gini, feature_index, threshold

    return best_feature, best_threshold


# Build Decision Tree Classifier
def build_tree(X, y, depth=0, max_depth=1):
    if len(y) == 0 or len(set(y)) == 1 or depth >= max_depth:
        return Node(value=np.sign(np.sum(y)))

    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=np.sign(np.sum(y)))

    X_left, X_right, y_left, y_right = partition(X, y, feature, threshold)
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth)

    return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)


# Predict function
def predict(node, X):
    if node.value is not None:
        return node.value
    if X[node.feature] <= node.threshold:
        return predict(node.left, X)
    else:
        return predict(node.right, X)


# Predict batch function
def predict_batch(tree, X):
    return np.array([predict(tree, sample) for sample in X])


# AdaBoost Classifier
def adaboost(X, y, n_estimators=50):
    n_samples = X.shape[0]
    weights = np.ones(n_samples) / n_samples #Divide by samples to create a distribution
    estimators = []
    alphas = []

    for _ in range(n_estimators):
        tree = build_tree(X, y, max_depth=1)
        predictions = predict_batch(tree, X)
        error = np.sum(weights * (predictions != y)) #Multiply by weights to focus more on important sample in next round

        if error > 0.5:
            continue

        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)

        estimators.append(tree)
        alphas.append(alpha)

    return estimators, alphas


# AdaBoost prediction
def adaboost_predict(estimators, alphas, X):
    final_prediction = np.zeros(X.shape[0])

    for alpha, tree in zip(alphas, estimators):
        final_prediction += alpha * predict_batch(tree, X)

    return np.sign(final_prediction)


# Main function
def main():
    X, y = load_data()
    estimators, alphas = adaboost(X, y, n_estimators=50)
    predictions = adaboost_predict(estimators, alphas, X)
    accuracy = np.mean(predictions == y)
    print(f"AdaBoost Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    main()

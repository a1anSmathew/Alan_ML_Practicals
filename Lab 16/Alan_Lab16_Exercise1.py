import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 1. Create sample regression data
X, y = make_regression(n_samples=200, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train multiple decision tree regressors
n_trees = 5
trees = []

for i in range(n_trees):
    tree = DecisionTreeRegressor(random_state=i)
    tree.fit(X_train, y_train)
    trees.append(tree) #Appends the entire model to the empty list

# 3. Aggregate predictions from all trees (average)
def aggregate_predictions(trees, X):
    # Each tree makes predictions
    predictions = np.array([tree.predict(X) for tree in trees])
    # Average the predictions across all trees
    return np.mean(predictions, axis=0)

# 4. Make final prediction on test set
final_predictions = aggregate_predictions(trees, X_test)

# 5. Show a few predictions
for i in range(5):
    print(f"Predicted: {final_predictions[i]:.2f}, Actual: {y_test[i]:.2f}")

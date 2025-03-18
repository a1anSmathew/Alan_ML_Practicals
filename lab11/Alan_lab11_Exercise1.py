import math
import numpy as np
import pandas as pd
from numpy.ma.extras import unique
from sklearn import datasets


def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y

    return X,y

class node:
    def __init__(self,feature=None, threshold=None,left=None,right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

#Function to compute the entropy
def entropy(y):
    entropy_value = 0
    unique_classes , counts = np.unique(y,return_counts=True) #First we find the unique values and the counts corresponding to it
    probabilities = counts/len(y) #probabilities is a list which contains the counts
    for p in probabilities: #Calculating the entropy
        if p > 0:
            entropy_value += -p * math.log2(p)
    return entropy_value

#Function for Partition
def partition(X,y,feature_index,threshold):
    left_mask = X[:,feature_index] <= threshold #left mask and right mask will contain boolean expressions for the row "feature_index" for the given condition
    right_mask = X[:,feature_index] > threshold
    X_left = X[left_mask] #All the rows in X which has "True" value in "feature_index" column for the left mask will be selected
    X_right = X[right_mask] #All the rows in X which has "True" value in "feature_index" column for the right mask will be selected
    y_left = y[left_mask] #All the rows in y for which there was corresponding true values in x will be selected
    y_right = y[right_mask]
    return X_left , X_right , y_left , y_right

#Computing the information gain
def Information_gain(y,y_left,y_right):
    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    weight_left = len(y_left)/len(y)
    weight_right = len(y_right)/len(y)
    H_children = (weight_left * H_left) + (weight_right * H_right)
    IG = H_parent - H_children

    return IG



#Function to decide best split
def best_split(X,y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    rows , columns = X.shape

    for feature_index in range (columns): #Looping through each feature to find the best feature
        sorted_indices = np.argsort(X[:,feature_index]) #Sorting the indices of the feature values in ascending order
        for j in range (rows - 1):
            threshold = (X[sorted_indices[j], feature_index] + X[sorted_indices[j + 1], feature_index]) / 2
            X_left,X_right,y_left,y_right = partition(X,y,feature_index,threshold)
            if len(y_left) == 0 or len(y_right) == 0: #If the threshold yields only one-sided tree, then we skip the threshold
                continue
            else:
                gain = Information_gain(y,y_left,y_right) #Calculation of information gain
            if gain > best_gain:
                best_gain , best_feature , best_threshold = gain , feature_index , threshold
    return best_feature , best_threshold


#Function for building a decision tree
def build_tree(X,y,depth=0,max_depth=5):
    # 1.To check if the stopping conditions are met
    unique_classes = np.unique(y) # Checking if all the points in the leaf/node belong to the same class
    if len(unique_classes) == 1 or depth >= max_depth: #If all the points in the node/leaf belong to the same class we stop building a tree for that node
        return node(value = np.bincount(y).argmax())

    # 2. Find the best feature and threshold
    feature , threshold = best_split(X, y)

    # 3. If no valid split is found, return the node value
    if feature is None: # If no feature is found good, there will be only 1 leaf
        return node(value = np.bincount(y).argmax())  # and the leaf will contain

    # 4. Partition data into left and right split
    X_left , X_right , y_left , y_right = partition(X , y , feature , threshold)

    # Step 5: Recursively build left and right subtrees
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth)

    # Step 6: Return a node with the selected feature and threshold
    return node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

def predict(node, X):
    if node.value is not None: #Only leaf node will have values
        return node.value
    if X[node.feature] <= node.threshold:
        return predict(node.left, X)
    else:
        return predict(node.right, X)

# Function to classify multiple samples
def predict_batch(tree, X):
    return np.array([predict(tree, sample) for sample in X]) #Loops through each row in the sample and carries out predict function for each sample


# Function to visualize the tree
def print_tree(node, depth=0):
    if node.value is not None:
        print(f"{'  ' * depth}Leaf: Class {node.value}")
    else:
        print(f"{'  ' * depth}Feature {node.feature} <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)


if __name__ == "__main__":
    X, y = load_data()

    # Train the decision tree
    tree = build_tree(X, y, max_depth=5)

    # Test on training data
    predictions = predict_batch(tree, X)
    accuracy = np.mean(predictions == y) * 100
    print("Decision Tree Accuracy:", accuracy, "%")

    # Print the tree structure
    print_tree(tree)







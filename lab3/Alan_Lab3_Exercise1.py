from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#Assigning the features and Ground truth
simulated_data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
print(simulated_data.columns)
y=simulated_data["disease_score"]
# y=simulated_data["disease_score_fluct"]
X=simulated_data.iloc[:, 0:-2]

#Exploratory Data Analysis
print(simulated_data.columns)
print(simulated_data)
print(simulated_data.info)
print(simulated_data.dtypes)
print(simulated_data.head(5))
# print(simulated_data.data.head())
# print(simulated_data.target.head())
# print(simulated_data.frame.info())


#Splitting the data into training and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

# Scale the data(Standardizing the data)
#We are scaling the data because the values in each column are in different scales (Eg. Gender is given as 1 and 0). If we don't scale it will also assume that columns with higher numbers will have high weightage
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Graphs for EDA
# 1. Histogram
simulated_data_graph = simulated_data.iloc[:,0:-2]
simulated_data_graph.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
plt.legend()
plt.show()

# 2 Scatter Plot
sns.scatterplot(
    data=simulated_data_graph,
    x="BP",
    y="blood_sugar",
    size="age",
    hue="age",
    palette="viridis",
    alpha=0.5,
)
plt.legend(title="BMI", bbox_to_anchor=(1.05, 0.95), loc="upper left")
_ = plt.title("BP and Blood Pressure correlation with gender")
plt.show()

# 3 Scatter Plot

pd.plotting.scatter_matrix(simulated_data[["age","disease_score"]])
plt.show()

plt.scatter(simulated_data["age"], simulated_data["disease_score"], color="blue", alpha=0.7)

# Add labels and title
plt.xlabel("Age")
plt.ylabel("Disease Score")
plt.title("Scatter Plot of Age vs Disease Score")

# Display the plot
plt.show()


rng = np.random.RandomState(0)
indices = rng.choice(
    np.arange(simulated_data_graph.shape[0]), size=500, replace=True
)

sns.scatterplot(
    data=simulated_data_graph.iloc[indices],
    x="BP",
    y="blood_sugar",
    size="age",
    hue="age",
    palette="viridis",
    alpha=0.5,
)
plt.legend(title="BMI", bbox_to_anchor=(1.05, 0.95), loc="upper left")
_ = plt.title("BP and Blood Pressure correlation with gender")
plt.show()

# Drop the unwanted columns
columns_drop = ["Gender"]
subset = simulated_data.iloc[indices].drop(columns=columns_drop)
# Quantize the target and keep the midpoint for each interval
subset["age"] = pd.qcut(subset["age"], 6, retbins=False)
subset["age"] = subset["age"].apply(lambda x: x.mid)

_ = sns.pairplot(data=subset, hue="age", palette="viridis")
plt.show()



# simulated_data["age"] = pd.qcut(simulated_data["age"], 6, retbins=False)
# simulated_data["age"] = simulated_data["age"].apply(lambda x: x.mid)
#
# _ = sns.pairplot(data=simulated_data, hue="age", palette="viridis")
# plt.show()

#Because age is a classification(1 and 0) hence qcut isn't required
_ = sns.pairplot(data=simulated_data, hue="Gender", palette="viridis")
plt.show()


# Training a linear regression
model: LinearRegression = LinearRegression()

# Training the mode
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Compute the r2 score
r2 = r2_score(y_test, y_pred)
print("r2 score is %0.2f (closer to 1 is good) " % r2)


import pandas as pd


def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.iloc[:,:-2]
    y = data.iloc[:,-2]

    return X,y


# def build_tree(X,y,t):
#     I1 = []
#     I2 = []
#     for index, j in X.iloc[:,2].items():
#         if j > t:
#             ind = X.index[index]
#             I1.append(ind)
#         else:
#             ind = X.index[index]
#             I2.append(ind)
#     print("I+ :",I1)
#     print("I- :", I2)
#
def build_tree(X, y,t):
    I1 = X[X.iloc[:, 2] > t].index.tolist()
    I2 = X[X.iloc[:, 2] <= t].index.tolist()

    print("I1:", I1)
    print("I2:", I2)

def main():
    X,y = load_data()
    t = 82
    build_tree(X,y,t)

if __name__ == '__main__':
    main()
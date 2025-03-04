import pandas as pd

#one-hot and ordinal encoding from scratch.
def one_hot():
    headers = ["Age", "Menopause", "Tumor_Size", "Inv_Nodes", "Node_Caps",
               "Deg_Malig", "Breast", "Breast_Quad", "Irradiat", "Class"]
    data = pd.read_csv("breastcancer.csv", header=None, names=headers)

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    categories = categorical_cols.to_list()
    print("Categorical Columns:", categories)

    for col in categories:
        unique_vals = data[col].unique()
        for val in unique_vals:
            data[f"{col}_{val}"] = data[col].apply(lambda x: 1 if x == val else 0)

    data = data.drop(columns=categorical_cols)

    print(data)

one_hot()

def ordinal():
    headers = ["Age", "Menopause", "Tumor_Size", "Inv_Nodes", "Node_Caps",
               "Deg_Malig", "Breast", "Breast_Quad", "Irradiat", "Class"]
    data = pd.read_csv("breastcancer.csv", header=None, names=headers)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    categories = categorical_cols.to_list()

    for col in categories:
        i = 0
        visited = []
        mapped_values = []
        for val in data[col]:
            if val not in visited:
                visited.append(val)
                mapped_values.append(i)
                i += 1
            else:
                mapped_values.append(visited.index(val))

        data[col] = mapped_values

ordinal()



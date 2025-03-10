import math

from fontTools.subset import subset


def ent_dict(data):
    entropy_dict = {}
    print("Total Number of Data Points: ",len(data))
    for i in data:
        if i not in entropy_dict.keys():
            entropy_dict[i] = 1
        else:
            entropy_dict[i] += 1
    print("Value Counts: ",entropy_dict)
    return entropy_dict

def entropy_calc(data,entropy_dict):
    entropy = 0
    val_list = list(entropy_dict.values())
    for i in range (len(val_list)) :
        p = val_list[i] / len(data)
        if p > 0:
            val = -p * math.log2(p)
            entropy += val
    print(entropy)
    return entropy

def Information_gain(data,subsets):
    parent_dict = ent_dict(data)
    parent_entropy = entropy_calc(data,parent_dict)

    total_entropy = 0
    for subset in subsets:
        subset_dict = ent_dict(subset)
        subset_entropy = -(len(subset)/len(data)) * entropy_calc(subset,subset_dict)
        total_entropy += subset_entropy

    I = parent_entropy - total_entropy
    return I


def main():
    data = ['A', 'B', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'A']
    subsets = [['A', 'A', 'A', 'A', 'A'],['B', 'B', 'B', 'B', 'B', 'B']]
    entropy_dict = ent_dict(data)
    entropy = entropy_calc(data,entropy_dict)
    I = Information_gain(data, subsets)
    print("Information_gain = ",I)

if __name__ == '__main__':
    main()
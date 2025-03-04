from cmath import sqrt

def L1_norm(v1,v2):
    l1_norm = []
    for i,j in zip (v1,v2):
        val = (i - j) ** 2
        l1_norm.append(val)
    l1 =sum(l1_norm)
    l1 = l1 ** (1/2)
    print(l1)

def L2_norm(v1,v2):
    l2_norm = []
    for i,j in zip (v1,v2):
        val = (i - j)
        if val < 0:
            val = val * (-1)
        l2_norm.append(val)
    print(sum(l2_norm))


def main():
    v1 = [5,9,3]
    v2 = [3,16,7]
    L1_norm(v1,v2)
    L2_norm(v1,v2)

if __name__ == '__main__':
    main()
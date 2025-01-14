import numpy as np
import random
d=4
E=0
k=1
n=random.randint(4,6)
while k<=n:
    y1 = 0
    for i in range (d):
        # np.random.seed(100)
        theta=random.randint(1,5)
        x=random.randint(1,5)
        sum=theta*x
        y1 += sum
    print("y1 value is: ",y1)

    y=random.randint(4,10)
    print("y",y)
    E += (y1-y)**2
    print("Error: ",E)

    k += 1
print("Final Error: ",E/2)




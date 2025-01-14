import numpy as np
A=[[1,2,3],[4,5,6]]
B=np.transpose(A)
print(A)
print(B)
C=B@A
print(C)
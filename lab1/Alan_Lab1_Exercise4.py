import numpy as np
import matplotlib.pyplot as plt
s=15
m=0
e=2.718
x=np.linspace(-100,100,100)
y=(1/(s*(2*3.14)**(-1/2)))*e**((-1/2)*((x-m)/s)**2)
plt.plot(x,y)
plt.show()
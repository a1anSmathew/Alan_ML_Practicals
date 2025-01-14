import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-100,100,100)
y=x**2
z=np.gradient(y,x)
plt.plot(x,z)
plt.plot(x,y)
plt.show()
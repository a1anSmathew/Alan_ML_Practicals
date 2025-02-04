from matplotlib import pyplot as plt

x_list = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
A_list = []
for x in x_list:
    gz = 1/(1 + (2.781**x))
    dA = gz * (1 - gz)
    A_list.append(dA)
plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(x_list, A_list, label="Derivative of Sigmoid Function", color="blue", linestyle="--", marker="o")

# Beautify the plot
plt.title("Derivative of Sigmoid Function", fontsize=16, fontweight='bold')
plt.xlabel("X-axis (Input)", fontsize=12)
plt.ylabel("Y-axis (Output)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper left", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()  # Adjust spacing for better layout

# Show the plot
plt.show()
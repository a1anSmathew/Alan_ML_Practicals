import matplotlib.pyplot as plt
import numpy as np
from mpmath.math2 import sqrt
def plotting(x1,x2,labels):
    colors = ['blue' if label == 'Blue' else 'red' for label in labels]

    plt.scatter(x1, x2, c=colors)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dot Plot of x1 vs x2 (colored by label)')
    plt.show()

def transform(x1,x2):
    x1_sqr = [i**2 for i in x1]
    x1_x2 = [round(sqrt(2)*i*j,2) for i,j in zip(x1,x2)]
    x2_sqr = [i**2 for i in x2]
    return(x1_sqr,x1_x2,x2_sqr)

def plotting2(x1_sqr,x1_x2,x2_sqr,labels):
    colors = ['blue' if label == 'Blue' else 'red' for label in labels]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_sqr, x1_x2, x2_sqr, c=colors)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('sqrt(2) * x1 * x2')
    ax.set_title('3D Scatter Plot with Non-linear Feature for Separation')
    plt.show()

    x_range = np.linspace(min(x1_sqr), max(x2_sqr), 10)
    y_range = np.linspace(min(x2_sqr), max(x2_sqr), 10)
    X, Y = np.meshgrid(x_range, y_range)

    # Define the plane: let's assume x3 = x1 + x2 (simple choice)
    Z = X + Y

    # Plot the plane
    ax.plot_surface(X, Y, Z, alpha=0.3, color='green')


def Question1(a):
    x1 = a[0]**2
    x2 = sqrt(2)*a[0]*a[1]
    x3 = a[1]**2

    a_transformed = [x1,x2,x3]
    return a_transformed

def dot_prod(a,b):
    return np.dot(a,b)

def polynomial_kernel(a,b):
    K = a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2
    return K

def main():
    x1 = [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16]
    x2 = [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3]
    labels = ['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue',
             'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red']
    x1_sqr , x1_x2 , x2_sqr = transform(x1,x2)
    plotting(x1,x2,labels)
    plotting2(x1_sqr, x1_x2, x2_sqr, labels)

    a = [3,6]
    b = [10,10]

    a_trans = Question1(a)
    b_trans = Question1(b)

    dp = dot_prod(a_trans,b_trans)
    print("dot product of vector a,b is: ",dp)

    K = polynomial_kernel(a,b)
    print("Value of the kernel function is: ", K)

if __name__ == '__main__':
    main()

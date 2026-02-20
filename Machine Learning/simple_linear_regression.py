import numpy as np
import matplotlib.pyplot as plt

# y = b0 + b1x
# Estimating Coefficient(a and b)
def estimate_coefficient(x,y):
    n = np.size(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    SS_xy = np.sum(y*x) - n*mean_x*mean_y # Sum of squared deviation between x and y 
    # SS_xy = ∑ (xi - mx)(yi - my) = ∑ xiyi - n*mx*my
    SS_xx = np.sum(x*x) - n*mean_x*mean_x # Sum of squared deviation of x
    # SS_xx = ∑ (xi - mx)^2 = ∑ xixi - n*mx*mx
    
    b1 = SS_xy/SS_xx
    b0 = mean_y - b1*mean_x
    
    return (b0, b1)

def regression_line(x, y, b):
    plt.scatter(x, y, marker="o", s=40)
    y_pred = b[0] + b[1]*x # Predicted response vector
    plt.plot(x, y_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    
def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    
    b = estimate_coefficient(x, y)
    print("Estimated Coefficient:\nb0 = {}\nb1 = {}".format(b[0], b[1]))
    
    regression_line(x, y, b)
    plt.show()

if __name__ == "__main__":
    main()
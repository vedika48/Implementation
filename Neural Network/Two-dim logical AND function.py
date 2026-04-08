import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

print("Input Points (X):")
print(X)
print("\nOutputs (y):")
print(y)

# f(x) = w1*x1 + w2*x2 + b
w = np.array([1, 1])
b = -1.5

print("\nUsing linear function: f(x) = x1 + x2 - 1.5")
print("\nEvaluating each point:")
for i in range(len(X)):
    value = np.dot(w, X[i]) + b
    print(f"Point {X[i]} → f(x) = {value:.2f} → Class = {'1' if value >= 0 else '0'}")

print("\nObservation:")
print("All class 0 points give NEGATIVE values")
print("Class 1 point gives POSITIVE value")
print("Therefore, a single straight line separates them")

class0 = X[y == 0]
class1 = X[y == 1]

plt.scatter(class0[:, 0], class0[:, 1], label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1')

x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = 1.5 - x_vals
plt.plot(x_vals, y_vals, linestyle='--', label='Decision Boundary')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("AND Function is Linearly Separable")
plt.legend()
plt.grid()
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.show()
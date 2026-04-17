import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0,1,1,0])

# PART 1: XOR + CONVEX HULL
plt.figure(figsize=(5,5))

class0 = X[y_xor == 0]
class1 = X[y_xor == 1]

plt.scatter(class0[:,0], class0[:,1], label="Class 0")
plt.scatter(class1[:,0], class1[:,1], label="Class 1")

def draw_hull(points):
    if len(points) >= 3:
        hull = ConvexHull(points)
        for s in hull.simplices:
            plt.plot(points[s,0], points[s,1])
    elif len(points) == 2:
        plt.plot(points[:,0], points[:,1])
    else:
        pass

draw_hull(class0)
draw_hull(class1)

plt.title("XOR NOT Linearly Separable (Convex Hulls)")
plt.legend()
plt.grid()
plt.show()

# 3D LIFTING
Z = X[:,0] * X[:,1]

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], Z[i], marker='o' if y_xor[i]==0 else '^')

ax.set_title("XOR separable in 3D (z = x1*x2)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x1*x2")
plt.show()

# LOGIC FUNCTIONS
def OR(x1,x2): return int(x1 or x2)
def NAND(x1,x2): return int(not (x1 and x2))
def XOR(x1,x2): return int(x1 != x2)
def NXOR(x1,x2): return int(x1 == x2)

functions = {"OR": OR, "NAND": NAND, "XOR": XOR, "NXOR": NXOR}

fig = plt.figure(figsize=(10,8))

for i,(name,f) in enumerate(functions.items(),1):
    ax = fig.add_subplot(2,2,i, projection='3d')
    Z = np.array([f(x[0],x[1]) for x in X])
    
    for j in range(len(X)):
        ax.scatter(X[j,0], X[j,1], Z[j])
    
    ax.set_title(name)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("output")

plt.tight_layout()
plt.show()

# 2-LAYER NN
def AND(x1,x2): return int(x1 and x2)

def neural_xor(x1,x2):
    h1 = OR(x1,x2)
    h2 = NAND(x1,x2)
    return AND(h1,h2)

print("\nTruth Table (NN solving XOR):")
print("x1 x2 | XOR_pred")
for x1,x2 in X:
    print(x1, x2, "|", neural_xor(x1,x2))
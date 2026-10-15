import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# AND Gate Dataset
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Perceptron weights for AND
w = np.array([1, 1])
b = -1.5

# -----------------------------
# Decision Boundary
# -----------------------------
xx = np.linspace(-0.2, 1.2, 200)
yy = -(w[0] * xx + b) / w[1]

plt.figure(figsize=(6,6))

# Plot Class 0
plt.scatter(
    X[y==0,0],
    X[y==0,1],
    s=120,
    marker='o',
    label='Class 0'
)

# Plot Class 1
plt.scatter(
    X[y==1,0],
    X[y==1,1],
    s=150,
    marker='^',
    label='Class 1'
)

# Decision Boundary
plt.plot(xx, yy, linewidth=2, label='Decision Boundary')

plt.title("AND Gate Decision Boundary")
plt.xlabel("Input x₁")
plt.ylabel("Input x₂")

plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)

plt.xticks([0,1])
plt.yticks([0,1])

plt.grid(True)
plt.legend()

plt.savefig("AND_Decision_Boundary.png", dpi=300, bbox_inches="tight")
plt.show()
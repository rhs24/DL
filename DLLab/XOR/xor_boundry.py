import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Define the XOR Dataset
# ------------------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# ------------------------------------
# Build the MLP Model
# ------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ------------------------------------
# Compile the Model
# ------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------
# Train the Model
# ------------------------------------
model.fit(X, y, epochs=5000, verbose=0)

# ------------------------------------
# Create a Grid for Visualization
# ------------------------------------
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Predict for every point in the grid
Z = model.predict(grid, verbose=0)
Z = Z.reshape(xx.shape)

# ------------------------------------
# Plot Decision Boundary
# ------------------------------------
plt.figure(figsize=(7, 7))

# Background decision regions
plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.5)

# Decision boundary line
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Plot XOR data points
plt.scatter(
    [0, 1],
    [0, 1],
    color='red',
    marker='x',
    s=150,
    linewidths=3,
    label='Class 0'
)

plt.scatter(
    [0, 1],
    [1, 0],
    color='blue',
    marker='o',
    s=150,
    label='Class 1'
)

# Point labels
plt.text(0.03, 0.03, "(0,0)", fontsize=10)
plt.text(0.03, 1.03, "(0,1)", fontsize=10)
plt.text(1.03, 0.03, "(1,0)", fontsize=10)
plt.text(1.03, 1.03, "(1,1)", fontsize=10)

# Labels and title
plt.xlabel("Input Feature $x_1$", fontsize=12)
plt.ylabel("Input Feature $x_2$", fontsize=12)
plt.title(
    "Figure 4: Decision Boundary Learned by the MLP",
    fontsize=14,
    fontweight="bold"
)

# Legend
plt.legend(loc="upper right")

# Save figure
plt.savefig(
    "Figure_4_Decision_Boundary_Learned_by_MLP.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
import matplotlib.pyplot as plt

# XOR dataset
class0_x = [0, 1]
class0_y = [0, 1]

class1_x = [0, 1]
class1_y = [1, 0]

# Create figure
plt.figure(figsize=(6, 6))

# Plot Class 0 (Negative)
plt.scatter(class0_x, class0_y, color='red', marker='x',
            s=120, label='Class 0')

# Plot Class 1 (Positive)
plt.scatter(class1_x, class1_y, color='blue', marker='o',
            s=120, label='Class 1')

# Label each point
points = [(0,0), (0,1), (1,0), (1,1)]
for x, y in points:
    plt.text(x + 0.03, y + 0.03, f"({x}, {y})", fontsize=10)

# Cartesian axes through the origin
plt.axhline(0, color='black', linewidth=1.5)
plt.axvline(0, color='black', linewidth=1.5)

# Axis limits and ticks
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xticks([0, 1])
plt.yticks([0, 1])

# Labels and title
plt.xlabel("Input Feature $x_1$", fontsize=12)
plt.ylabel("Input Feature $x_2$", fontsize=12)
plt.title("Figure 1: XOR Dataset", fontsize=14, fontweight="bold")

# Remove grid
plt.grid(False)

# Legend
plt.legend(loc="center")

# Save the figure
plt.savefig(
    "Figure_1_XOR_Dataset.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Set Random Seed
# ---------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------
# XOR Dataset
# ---------------------------------
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

# ---------------------------------
# Build the MLP Model
# ---------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ---------------------------------
# Compile the Model
# ---------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# ---------------------------------
# Train the Model
# ---------------------------------
history = model.fit(
    X,
    y,
    epochs=5000,
    verbose=0
)

# ---------------------------------
# Evaluate the Model
# ---------------------------------
loss, accuracy = model.evaluate(X, y, verbose=0)

print("\nModel Performance")
print("-" * 30)
print(f"Final Loss     : {loss:.6f}")
print(f"Final Accuracy : {accuracy * 100:.2f}%")

# ---------------------------------
# Predictions
# ---------------------------------
predictions = model.predict(X, verbose=0)

print("\nPredictions")
print("-" * 45)
print("{:<12}{:<12}{:<12}{:<10}".format(
    "Input", "Expected", "Probability", "Class"))

for inp, actual, pred in zip(X, y, predictions):
    probability = pred.item()
    predicted_class = int(probability >= 0.5)

    print("{!s:<12}{:<12}{:<12.4f}{:<10}".format(
        inp,
        int(actual.item()),
        probability,
        predicted_class
    ))

# ---------------------------------
# Plot Training Loss
# ---------------------------------
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], linewidth=2)

plt.title("Figure 2: Training Loss for XOR Classifier",
          fontsize=14,
          fontweight='bold')

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")

plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(
    "Figure_2_Training_Loss_XOR.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()
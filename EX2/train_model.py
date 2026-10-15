import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Generate Dataset
# -------------------------------------------------
X, y = make_moons(
    n_samples=1000,
    noise=0.25,
    random_state=42
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Build Model
# -------------------------------------------------
def build_model(loss_function):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss=loss_function,
        metrics=['accuracy']
    )

    return model

# -------------------------------------------------
# Model A : MSE
# -------------------------------------------------

model_mse = build_model("mse")

history_mse = model_mse.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)

# -------------------------------------------------
# Model B : BCE
# -------------------------------------------------

model_bce = build_model("binary_crossentropy")

history_bce = model_bce.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

mse_loss, mse_acc = model_mse.evaluate(X_test, y_test, verbose=0)
bce_loss, bce_acc = model_bce.evaluate(X_test, y_test, verbose=0)

print("\nModel Performance\n")

print(f"MSE Accuracy : {mse_acc*100:.2f}%")
print(f"BCE Accuracy : {bce_acc*100:.2f}%")

# ===================================================
# Plot 1 : Training Loss
# ===================================================

plt.figure(figsize=(8,5))

plt.plot(history_mse.history['loss'], label="MSE")
plt.plot(history_bce.history['loss'], label="Binary Cross Entropy")

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("training_loss_comparison.png", dpi=300, bbox_inches="tight")

plt.show()


# ===================================================
# Plot 2 : Validation Accuracy
# ===================================================

plt.figure(figsize=(8,5))

plt.plot(history_mse.history['val_accuracy'], label="MSE")
plt.plot(history_bce.history['val_accuracy'], label="Binary Cross Entropy")

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("validation_accuracy_comparison.png", dpi=300, bbox_inches="tight")

plt.show()


# ===================================================
# Plot 3 : Convergence Speed
# ===================================================

plt.figure(figsize=(8,5))

plt.plot(history_mse.history['loss'], label="MSE")
plt.plot(history_bce.history['loss'], label="Binary Cross Entropy")

plt.xlim(0,30)

plt.title("Convergence Speed Comparison (First 30 Epochs)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("convergence_speed_comparison.png", dpi=300, bbox_inches="tight")

plt.show()
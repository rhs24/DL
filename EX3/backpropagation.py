import numpy as np

# Inputs
x = np.array([[1],[0]])
y = 1

# Initial weights
W1 = np.array([[0.2,0.3],
               [0.4,0.1]])

W2 = np.array([[0.5],
               [0.6]])

lr = 0.1


def sigmoid(x):
    return 1/(1+np.exp(-x))


# Forward propagation

z1 = W1.T @ x
h = sigmoid(z1)

z2 = W2.T @ h
y_hat = sigmoid(z2)


# Loss

loss = 0.5*(y-y_hat)**2


# Backpropagation

dL_dy = y_hat-y

delta2 = dL_dy*y_hat*(1-y_hat)

dW2 = h*delta2


delta1 = (W2*delta2)*(h*(1-h))

dW1 = x@delta1.T


# Weight update

W1 = W1-lr*dW1
W2 = W2-lr*dW2


print("Prediction:", y_hat)
print("Loss:", loss)

print("\nUpdated W1:")
print(W1)

print("\nUpdated W2:")
print(W2)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# region BUILD A MODEL
# generate 20 data points
N = 20
# random data on the x-axis in (-5, +5)
X = np.random.random(N) * 10 - 5
# a line plus some noise
Y = 0.5 * X - 1 + np.random.randn(N)
# plt.scatter(X, Y)
# plt.show()
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# num_samples x num_dimensions
X = X.reshape(N, 1)
Y = Y.reshape(N, 1)
inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))
# endregion

# region TRAIN THE MODEL
n_epochs = 30
losses = []
for it in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f'Epoch {it + 1}/{n_epochs}, Loss: {loss.item():.4f}')

# plt.plot(losses)
# plt.show()
# endregion

# region MAKE PREDICTION
predicted = model(inputs).detach().numpy()
w = model.weight.data.numpy()
b = model.bias.data.numpy()
print("weight and bias: ", w, b)
plt.scatter(X, Y, label='Original data')
plt.plot(X, predicted, label='Fitted Line')
plt.legend()
plt.show()

# endregion

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path("data/moore.csv")
data = pd.read_csv(data_path, header=None).values
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

# plt.scatter(X, Y)
# plt.show()

Y = np.log(Y)
mean_x = X.mean()
std_x = X.std()
mean_y = Y.mean()
std_y = Y.std()
X = (X - mean_x) / std_x
Y = (Y - mean_y) / std_y

X = X.astype(np.float32)
Y = Y.astype(np.float32)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
inputs = torch.from_numpy(X)
targets = torch.from_numpy(Y)

n_epochs = 100
losses = []
for it in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss)
    loss.backward()
    optimizer.step()
    print(f'Epoch {it + 1}/{n_epochs}, Loss: {loss.item():.4f}')

# plt.plot(losses)
# plt.show()

predicted = model(torch.from_numpy(X)).detach().numpy()
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, predicted, label='Fitted line')
plt.legend()
# plt.show()

w = model.weight.data.numpy()
b = model.bias.data.numpy()
print("weight and bias: ", w, b)
# recover original slope not normalized
# 2c = cr^t -> tt - t = log(2) / log(r) -> log(r) = a
a = w[0, 0] * std_y / std_x
print('Time to double:', np.log(2) / a)

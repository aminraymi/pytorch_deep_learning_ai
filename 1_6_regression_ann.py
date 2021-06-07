import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 1000
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2 * X[:, 0]) + np.cos(3 * X[:, 1])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], Y)
#plt.show()

model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def full_gd(model, criterion, optimizer, X_train, y_train, epochs=1000):
    train_losses = np.zeros(epochs)
    for it in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses[it] = loss.item()
        if (it + 1) % 50 == 0:
            print(f'epoch {it + 1}/{epochs}, Train loss: {loss.item():.4f}')

    return train_losses


X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(Y.astype(np.float32).reshape(-1, 1))
train_losses = full_gd(model, criterion, optimizer, X_train, y_train)
# plt.plot(train_losses)
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
with torch.no_grad():
    line = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(line, line)
    xgrid = np.vstack((xx.flatten(), yy.flatten())).T
    xgrid_torch = torch.from_numpy(xgrid.astype(np.float32))
    yhat = model(xgrid_torch).numpy().flatten()
    ax.plot_trisurf(xgrid[:,0], xgrid[:, 1], yhat, linewidth=0.2, antialiased=True)
    plt.show()



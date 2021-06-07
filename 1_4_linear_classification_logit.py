import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
print(data.keys())
print(data.data.shape)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# we don't fit this to be surprise
X_test = scaler.transform(X_test)

model = nn.Sequential(
    nn.Linear(D, 1)
)

# BCE with logit + sigmoid -> remove sigmoid from nn.Linear
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

n_epochs = 1000
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

for it in range(n_epochs):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)

    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()

    if (it + 1) % 50 == 0:
        print(f'epoch {it + 1}/{n_epochs}, Train loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

"""
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()
"""

# calculate accuracy
with torch.no_grad():
    p_train = model(X_train)
    p_train = np.round(p_train.numpy())
    train_accuracy = np.mean(y_train.numpy() == p_train)

    p_test = model(X_test)
    p_test = np.round(p_test.numpy())
    test_accuracy = np.mean(y_test.numpy() == p_test)

print(f'Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}')

# save model
print(model.state_dict())
torch.save(model.state_dict(), "model.pt")

# load model
model_two = nn.Sequential(
    nn.Linear(D, 1),
    nn.Sigmoid()
)
model_two.load_state_dict(torch.load('model.pt'))

# evaluate the new model
with torch.no_grad():
    p_train = model_two(X_train)
    p_train = np.round(p_train.numpy())
    train_accuracy = np.mean(y_train.numpy() == p_train)

    p_test = model_two(X_test)
    p_test = np.round(p_test.numpy())
    test_accuracy = np.mean(y_test.numpy() == p_test)

print(f'Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}')

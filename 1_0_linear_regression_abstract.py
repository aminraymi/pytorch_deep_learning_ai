import torch

# 1. build a model
model = torch.nn.Linear(1, 1)

# 2. train a model
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# inputs = torch.from_numpy(X.astype(np.float32))
inputs = "null"
targets = "null"

n_epochs = 30
for it in range(n_epochs):
    # zero the parameter gradients

    optimizer.zero_grad()

    # forwards pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # backward and optimize
    loss.backward()
    optimizer.step()

# 3. make prediction
predictions = model(inputs).detach().numpy()

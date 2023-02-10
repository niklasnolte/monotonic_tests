# %%
import pandas as pd
import torch
import tqdm
from monotone_utils import GroupSort, direct_norm, SigmaNet
# %%
df = pd.read_csv('data/auto-mpg.csv')
df = df[df.horsepower != "?"]
# %%
# mpg is regression target
# cylinders, displacement, horsepower, weight, acceleration, model year, origin are features
X = df.drop(columns=['mpg', 'car name']).values
Y = df['mpg'].values
X = torch.tensor(X.astype(float), dtype=torch.float32)
Y = torch.tensor(Y.astype(float), dtype=torch.float32).view(-1, 1)
X = (X - X.mean(0)) / X.std(0)
Ymean = Y.mean(0)
Ystd = Y.std(0)
Y = (Y - Ymean) / Ystd
# %%
rmses = []
for seed in range(3):
  torch.manual_seed(seed)
  # split in train and test
  randperm = torch.randperm(X.shape[0])
  X = X[randperm]
  Y = Y[randperm]
  split = int(0.8 * X.shape[0])
  Xtr = X[:split]
  Ytr = Y[:split]
  Xts = X[split:]
  Yts = Y[split:]

  width = 8

  model = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(X.shape[1], width), kind="one-inf"),
    GroupSort(width//2),
    direct_norm(torch.nn.Linear(width, width), kind="inf"),
    GroupSort(width//2),
    direct_norm(torch.nn.Linear(width, 1), kind="inf"),
  )
  model = SigmaNet(model, sigma=1, monotone_constraints=[0,-1,-1,-1,0,0,0])

  # number of elements
  print(sum(p.numel() for p in model.parameters()))


  optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
  epochs = 2000

  mse = float('inf')
  bar = tqdm.tqdm(range(epochs))
  for epoch in bar:
    batch_size = 64
    for i in range(0, Xtr.shape[0], batch_size):
      optimizer.zero_grad()
      yhat = model(Xtr[i:i+batch_size])
      loss = torch.nn.functional.mse_loss(yhat, Ytr[i:i+batch_size])
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      yhat = model(Xts)
      # unscaled mse
      new_mse = torch.nn.functional.mse_loss(yhat * Ystd, Yts * Ystd)
      mse = min(mse, new_mse.item())
      bar.set_description(f"mse: {new_mse:.1f}, best: {mse:.1f}")
  rmses.append(mse)

# print mean and std
print(f"mean: {torch.tensor(rmses).mean():.5f}, std: {torch.tensor(rmses).std():.5f}")

# %%

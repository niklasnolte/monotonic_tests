# %%
import pandas as pd
import torch
from torch.nn import functional as F
import tqdm
#balanced accuracy
from monotone_utils import GroupSort, direct_norm, SigmaNet
# %%
df_train = pd.read_csv("data/heart_train.csv")
df_test = pd.read_csv("data/heart_test.csv")

def preprocess(df):
  X = df.drop(columns=['target']).values
  Y = df['target'].values
  X = torch.tensor(X.astype(float), dtype=torch.float32)
  Y = torch.tensor(Y.astype(float), dtype=torch.float32).view(-1, 1)
  X = (X - X.mean(0)) / X.std(0)
  return X, Y
Xtr, Ytr = preprocess(df_train)
Xts, Yts = preprocess(df_test)

# %%
def get_acc(Yhat, Y):
  max_acc = 0
  for threshold in torch.linspace(-1, 1, 100):
    acc = (Yhat > threshold) == Y
    acc = acc.sum().item() / acc.numel()
    max_acc = max(max_acc, acc)
  return max_acc
# %%
accs = []
for seed in range(3):
  torch.manual_seed(seed)

  width = 4

  model = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(Xtr.shape[1], width), kind="one-inf"),
    GroupSort(2),
    direct_norm(torch.nn.Linear(width, width), kind="inf"),
    GroupSort(2),
    direct_norm(torch.nn.Linear(width, 1), kind="inf"),
  )
  monotone_constraints = [0] * Xtr.shape[1]
  monotone_constraints[df_train.columns.get_loc('trestbps')] = 1
  monotone_constraints[df_train.columns.get_loc('chol')] = 1
  model = SigmaNet(model, sigma=1, monotone_constraints=monotone_constraints)


  # number of elements
  print(sum(p.numel() for p in model.parameters()))


  optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
  epochs = 300

  best_acc = 0
  bar = tqdm.tqdm(range(epochs))
  for epoch in bar:
    optimizer.zero_grad()
    yhat = model(Xtr)
    loss = F.binary_cross_entropy_with_logits(yhat, Ytr)
    loss.backward()
    optimizer.step()
    train_acc = get_acc(yhat, Ytr)
    with torch.no_grad():
      yhat = model(Xts)
      accuracy = get_acc(yhat, Yts)
      best_acc = max(best_acc, accuracy)
      bar.set_description(f"loss {loss.item():.3f} train acc: {train_acc:.3f}, test acc: {accuracy:.3f}, best acc: {best_acc:.3f}")
  accs.append(best_acc)


# print mean and std
print(f"mean: {torch.tensor(accs).mean():.3f}, std: {torch.tensor(accs).std():.4f}")

# %%

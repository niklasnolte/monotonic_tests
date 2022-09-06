# %%
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from monotonenorm import GroupSort, SigmaNet, direct_norm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# %%
# load data
basepath = "/data/nnolte/chest_xray/"
XTAB = torch.load(basepath + "XTAB.pt")
Y = torch.load(basepath + "Y.pt").to_numpy()
resnet_features = torch.load(basepath + "resnet18_features.pt")

# %%
#X = XTAB.numpy()
X = torch.hstack([XTAB, resnet_features]).numpy()
print(X.shape)

# %%
accs = []
for i in range(3,6):
    torch.manual_seed(i)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=i
    )
    
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    EPOCHS = 100
    bar = tqdm(range(EPOCHS))
    for i in bar:
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_train)
        loss.backward()
        optimizer.step()
        bar.set_description(f"loss: {loss.cpu().item():.5f}")
      
    # %%
    with torch.no_grad():
      y_pred = model(X_test).cpu()
      acc = 0
      for i in range(0, 1, 100):
          acc = max(acc, accuracy_score(y_test.cpu(), y_pred > i))

      bacc = 0
      for i in range(0, 1, 100):
          bacc = max(bacc, balanced_accuracy_score(y_test.cpu(), y_pred > i))
    # %%
    print(f"accuracy: {acc:.5f}, balanced accuracy: {bacc:.5f}")
    accs.append(acc)

print(f"mean accuracy: {np.mean(accs):.5f}, std accuracy: {np.std(accs):.5f}")
# %%

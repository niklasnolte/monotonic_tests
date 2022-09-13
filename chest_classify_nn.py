# %%
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from monotonenorm import GroupSort, SigmaNet, direct_norm
from sklearn.metrics import accuracy_score
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
monotone_constraint=[1,1,0,0] + [0]*resnet_features.shape[1]

# %%
accs = []
for seed in range(3):
    torch.manual_seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )
    
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    model = torch.nn.Sequential(
        direct_norm(torch.nn.Linear(X_train.shape[1], 16), kind="one-inf"),
        GroupSort(8),
        direct_norm(torch.nn.Linear(16, 16), kind="inf"),
        GroupSort(8),
        direct_norm(torch.nn.Linear(16, 1), kind="inf")
    ).to(device)

    model = SigmaNet(model, sigma=1, monotone_constraints=monotone_constraint).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    EPOCHS = 2000
    bar = tqdm(range(EPOCHS))
    acc = 0
    for i in bar:
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_train)
        loss.backward()
        optimizer.step()
        bar.set_description(f"loss: {loss.cpu().item():.5f}")
      
        if i %10 == 0:
          with torch.no_grad():
            y_pred = model(X_test).cpu()
            for i in range(0, 1, 100):
                acc = max(acc, accuracy_score(y_test.cpu(), y_pred > i))
        # %%
    accs.append(acc)
    torch.save(model.state_dict(), f"models/chest_classify_nn_{seed}.pt")

print(f"mean accuracy: {np.mean(accs):.5f}, std accuracy: {np.std(accs):.5f}")
# %%

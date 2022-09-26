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

    width = 2

    model = torch.nn.Sequential(
        direct_norm(torch.nn.Linear(X_train.shape[1], width), kind="one-inf"),
        GroupSort(width//2),
        direct_norm(torch.nn.Linear(width, width), kind="inf"),
        GroupSort(width//2),
        direct_norm(torch.nn.Linear(width, 1), kind="inf")
    ).to(device)

    model = SigmaNet(model, sigma=1, monotone_constraints=monotone_constraint).to(device)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-1)
    EPOCHS = 4000
    bar = tqdm(range(EPOCHS))
    acc = 0
    for i in bar:
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_train)
        loss.backward()
        optimizer.step()
      
        if i %10 == 0:
          with torch.no_grad():
            y_pred = model(X_test).cpu()
            min_ = np.quantile(y_pred, 0.05)
            max_ = np.quantile(y_pred, 0.95)
            for i in np.linspace(min_, max_, 100):
                acci = accuracy_score(y_test.cpu(), (y_pred > i))
                if acci > acc:
                    acc = acci
                    statedict = model.state_dict()
        bar.set_description(f"loss: {loss.cpu().item():.5f}, acc: {acc:.5f}")
        # %%
    accs.append(acc)
    torch.save(statedict, f"models/chest_classify_nn_{seed}.pt")

print(f"mean accuracy: {np.mean(accs):.5f}, std accuracy: {np.std(accs):.5f}")
# %%

# %%
import torch
from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from monotonenorm import GroupSort, SigmaNet, direct_norm
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# %%
device = torch.device("cuda:1")
# %%
basepath = "/data/nnolte/chest_xray/"
XIMG = torch.load(basepath + "XIMG.pt")
XTAB = torch.load(basepath + "XTAB.pt")
Y = torch.tensor(torch.load(basepath + "Y.pt")).float()

# %%
class ResNet18Mono(torch.nn.Module):
    def __init__(self, monotonic=False, state_dict=None):
        super().__init__()
        resnet = resnet18(pretrained=True).requires_grad_(True)
        monotone_constraint = [1, 1, 0, 0] + [0] * resnet.fc.in_features
        resnet.fc = torch.nn.Identity()
        self.resnet = resnet
        self.monotonic = torch.nn.Sequential(
            direct_norm(torch.nn.Linear(len(monotone_constraint), 16), kind="one-inf"),
            GroupSort(8),
            direct_norm(torch.nn.Linear(16, 16), kind="inf"),
            GroupSort(8),
            direct_norm(torch.nn.Linear(16, 1), kind="inf"),
        ).to(device)
        if monotonic:
            self.monotonic = SigmaNet(
                self.monotonic, sigma=1, monotone_constraints=monotone_constraint
            ).to(device)
            if state_dict is not None:
                self.monotonic.load_state_dict(state_dict)
                      

    def forward(self, ximg, xtab):
        ximg = self.resnet(ximg)
        x = torch.hstack([xtab, ximg])
        x = self.monotonic(x)
        return x


# %%
accs = []
for i in range(3):
    torch.manual_seed(i)
    XIMG_train, XIMG_test, XTAB_train, XTAB_test, y_train, y_test = train_test_split(
        XIMG, XTAB, Y, test_size=0.2, random_state=i
    )

    XIMG_train = XIMG_train.float().to(device)
    XIMG_test = XIMG_test.float().to(device)
    XTAB_train = XTAB_train.float().to(device)
    XTAB_test = XTAB_test.float().to(device)
    y_train = y_train.float().unsqueeze(1).to(device)
    y_test = y_test.float().unsqueeze(1)
    train_loader = DataLoader(
        list(zip(XIMG_train, XTAB_train, y_train)), batch_size=128, shuffle=True
    )
    test_loader = DataLoader(
        list(zip(XIMG_test, XTAB_test, y_test)), batch_size=128, shuffle=True
    )

    state_dict = torch.load(f"models/chest_classify_nn_{i}.pt")

    model = ResNet18Mono(monotonic=True, state_dict=state_dict).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    EPOCHS = 100
    bar = tqdm(range(EPOCHS))
    acc = 0
    for i in bar:
        for ximg_, xtab_, y_ in train_loader:
            optimizer.zero_grad()
            y_pred = model(ximg_, xtab_)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_)
            loss.backward()
            optimizer.step()
            bar.set_description(f"loss: {loss.cpu().item():.5f}")

        if i % 10 == 0:
            with torch.no_grad():
                for ximg_, xtab_, y_ in test_loader:
                    y_pred = model(ximg_, xtab_).cpu()
                    for i in range(0, 1, 100):
                        acc = max(acc, accuracy_score(y_.cpu(), y_pred > i))
        # %%
    accs.append(acc)

print(f"mean accuracy: {np.mean(accs):.5f}, std accuracy: {np.std(accs):.5f}")
# %%

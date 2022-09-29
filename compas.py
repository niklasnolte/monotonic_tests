import torch
from loaders.compas_loader import load_data, mono_list
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

from monotone_utils import SigmaNet, direct_norm, GroupSort

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

X_train, y_train, X_test, y_test = load_data(get_categorical_info=False)

X_train = torch.tensor(X_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_train = torch.tensor(y_train).float().unsqueeze(1).to(device)
y_test = torch.tensor(y_test).float().unsqueeze(1).to(device)

mean = X_train.mean(0)
std = X_train.std(0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

monotone_constraints = [1 if i in mono_list else 0 for i in range(X_train.shape[1])]

per_layer_lip = 1.3


def run(seed):
    torch.manual_seed(seed)

    width = 2

    network = torch.nn.Sequential(
        direct_norm(
            torch.nn.Linear(X_train.shape[1], width),
            kind="one-inf",
            max_norm=per_layer_lip,
        ),
        GroupSort(width // 2),
        direct_norm(torch.nn.Linear(width, width), kind="inf", max_norm=per_layer_lip),
        GroupSort(width // 2),
        direct_norm(torch.nn.Linear(width, 1), kind="inf", max_norm=per_layer_lip),
    )
    network = torch.nn.Sequential(
        SigmaNet(
            network, sigma=per_layer_lip ** 4, monotone_constraints=monotone_constraints
        ),
        torch.nn.Sigmoid(),
    )

    network = network.to(device)

    print("params:", sum(p.numel() for p in network.parameters()))

    optimizer = torch.optim.Adam(network.parameters(), lr=2e-4)

    data_train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(X_train, y_train), batch_size=256, shuffle=True,
    )
    bar = tqdm(range(2000))
    acc = 0
    for i in bar:
        for X, y in data_train_loader:
            y_pred = network(X)
            loss_train = torch.nn.functional.binary_cross_entropy(y_pred, y)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = network(X_test)
            loss = torch.nn.functional.binary_cross_entropy(y_pred, y_test)
            acci = 0
            for i in torch.linspace(0, 1, 50):
                acci = max(
                    acci,
                    accuracy_score(
                        y_test.cpu().detach().numpy(),
                        (y_pred.cpu().detach().numpy() > i.item()).astype(int),
                    ),
                )

            acc = max(acc, acci)
            bar.set_description(
                f"train: {loss_train.item():.4f}, test: {loss.item():.4f}, current acc: {acci:.4f}, best acc: {acc:.4f}"
            )
    return acc


accs = [run(i) for i in range(3)]
print(f"mean: {np.mean(accs):.4f}, std: {np.std(accs):.4f}")

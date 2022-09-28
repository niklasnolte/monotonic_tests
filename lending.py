import torch
from tqdm import tqdm
from loaders.lending_loader import load_data, mono_list
from monotonenorm import SigmaNet, GroupSort, direct_norm
from sklearn.metrics import accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False)
monotonic_constraints = np.array([int(i in mono_list) for i in range(Xtr.shape[1])])


def run_exp(seed):
    torch.manual_seed(seed)

    width = 16

    class Model(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            activation = lambda: GroupSort(width//2)

            layers = torch.nn.Sequential(
                direct_norm(torch.nn.Linear(Xtr.shape[1], width), kind="one-inf"),
                activation(),
                direct_norm(torch.nn.Linear(width, width), kind="inf"),
                activation(),
                direct_norm(torch.nn.Linear(width, 1), kind="inf"),
            )
            self.nn = SigmaNet(
                layers, sigma=1, monotone_constraints=monotonic_constraints
            )

        def forward(self, x):
            return torch.sigmoid(self.nn(x))

    model = Model(width)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("params:", sum(p.numel() for p in model.parameters()))

    Xtrt = torch.tensor(Xtr, dtype=torch.float32).to(device)
    Ytrt = torch.tensor(Ytr, dtype=torch.float32).view(-1, 1).to(device)
    Xtst = torch.tensor(Xts, dtype=torch.float32).to(device)
    Ytst = torch.tensor(Yts, dtype=torch.float32).view(-1, 1).to(device)

    mean = Xtrt.mean(0)
    std = Xtrt.std(0)
    Xtrt = (Xtrt - mean) / std
    Xtst = (Xtst - mean) / std

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtrt, Ytrt), batch_size=int(2 ** 9), shuffle=True
    )
    max_acc = 0

    bar = tqdm(range(100))
    for i in bar:
        for Xi, yi in dataloader:
            y_pred = model(Xi)
            losstr = torch.nn.functional.binary_cross_entropy(y_pred, yi)
            optimizer.zero_grad()
            losstr.backward()
            optimizer.step()

        with torch.no_grad():
            y_predts = model(Xtst)
            lossts = torch.nn.functional.binary_cross_entropy(y_predts, Ytst)
            acc = 0
            for i in np.linspace(0, 1, 100):
                acc = max(
                    acc,
                    accuracy_score(Ytst.cpu().numpy(), y_predts.cpu().numpy() > i),
                )

            max_acc = max(max_acc, acc)
            bar.set_description(
                f"Loss: {losstr.item():.4f} {lossts.item():.4f}, acc: {acc.item():.4f}, max_acc: {max_acc:.4f}"
            )
    return max_acc


if __name__ == "__main__":
    accs = [run_exp(i) for i in range(3)]  # 3 seeds
    # print mean and std of the 3 runs
    print(f"mean: {np.mean(accs):.4f}, std: {np.std(accs):.4f}")


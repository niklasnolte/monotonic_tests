import torch
from monotone_utils import SigmaNet, GroupSort, direct_norm
from tqdm import tqdm

def run_exp(
    Xtr, Ytr, Xts, Yts, monotone_constraints,
    max_lr, expwidth, depth, Lip, batchsize, seed
):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xtrt = torch.tensor(Xtr, dtype=torch.float32).to(device)
    Ytrt = torch.tensor(Ytr, dtype=torch.float32).view(-1, 1).to(device)
    Xtst = torch.tensor(Xts, dtype=torch.float32).to(device)
    Ytst = torch.tensor(Yts, dtype=torch.float32).view(-1, 1).to(device)

    # normalize training data
    mean = Xtrt.mean(0)
    std = Xtrt.std(0)
    Xtrt = (Xtrt - mean) / std
    Xtst = (Xtst - mean) / std

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtrt, Ytrt), batch_size=batchsize, shuffle=True
    )

    per_layer_lip = Lip ** (1 / depth)
    width = 2 ** expwidth

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            activation = lambda: GroupSort(width // 2)

            layers = [
                direct_norm(
                    torch.nn.Linear(Xtrt.shape[1], width),
                    kind="one-inf",
                    max_norm=per_layer_lip,
                ),
                activation(),
            ]
            for _ in range(depth - 2):
                layers.append(
                    direct_norm(
                        torch.nn.Linear(width, width),
                        kind="inf",
                        max_norm=per_layer_lip,
                    )
                )
                layers.append(activation())

            layers.append(
                direct_norm(
                    torch.nn.Linear(width, 1), kind="inf", max_norm=per_layer_lip
                )
            )

            self.nn = SigmaNet(
                torch.nn.Sequential(*layers),
                sigma=Lip,
                monotone_constraints=monotone_constraints,
            )

        def forward(self, x):
            return self.nn(x)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    EPOCHS = 1000

    print("params:", sum(p.numel() for p in model.parameters()))
    bar = tqdm(range(EPOCHS))
    best_rmse = 1
    for _ in bar:
        model.train()
        for Xi, yi in dataloader:
            y_pred = model(Xi)
            losstr = torch.nn.functional.mse_loss(y_pred, yi)
            optimizer.zero_grad()
            losstr.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_predts = model(Xtst)
            lossts = torch.nn.functional.mse_loss(y_predts, Ytst)
            tsrmse = lossts.item() ** 0.5
            trrmse = losstr.item() ** 0.5
            best_rmse = min(best_rmse, tsrmse)
            bar.set_description(
                f"train rmse: {trrmse:.5f} test rmse: {tsrmse:.5f}, best: {best_rmse:.5f}, lr: {optimizer.param_groups[0]['lr']:.5f}"
            )
    return best_rmse

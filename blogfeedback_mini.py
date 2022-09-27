# %%
import torch
from tqdm import tqdm
from loaders.blog_loader import load_data, mono_list
from monotonenorm import SigmaNet, GroupSort
import numpy as np
from sklearn.linear_model import Ridge

# %%
Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False)
input_dim = Xtr.shape[1]
monotone_constraints = np.array([1 if i in mono_list else 0 for i in range(input_dim)])

print(Xtr.shape, Ytr.shape, Xts.shape, Yts.shape)

model = Ridge()
model.fit(Xtr, Ytr, )
rmse = np.sqrt(np.mean((model.predict(Xts) - Yts)**2))
print(rmse)
important_feature_idxs = np.argsort(model.coef_)[::-1][:20]


def run_exp(
    max_lr=2e-3, expwidth=2, depth=4, Lip=1, monotonic=True, batchsize=256, seed=1, gpu_id=0
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xtrt = torch.tensor(Xtr[:, important_feature_idxs], dtype=torch.float32).to(device)
    Ytrt = torch.tensor(Ytr, dtype=torch.float32).view(-1, 1).to(device)
    Xtst = torch.tensor(Xts[:, important_feature_idxs], dtype=torch.float32).to(device)
    Ytst = torch.tensor(Yts, dtype=torch.float32).view(-1, 1).to(device)

    # normalize training data
    mean = Xtrt.mean(0)
    std = Xtrt.std(0)
    Xtrt = (Xtrt - mean) / std
    Xtst = (Xtst - mean) / std


    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtrt, Ytrt), batch_size=batchsize
    )

    per_layer_lip = Lip ** (1 / depth)
    width = 2 ** expwidth

    class Model(torch.nn.Module):
        def __init__(self, robust=False, sigma=False):
            if sigma and not robust:
                raise ValueError("sigma requires robust")
            super().__init__()
            activation = lambda: GroupSort(width // 2)
            if robust:
                from monotonenorm import direct_norm
            else:
                direct_norm = lambda x, *args, **kwargs: x  # make it a normal network

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
                        torch.nn.Linear(width, width), kind="inf", max_norm=per_layer_lip
                    )
                )
                layers.append(activation())

            layers.append(
                direct_norm(torch.nn.Linear(width, 1), kind="inf", max_norm=per_layer_lip)
            )

            self.nn = torch.nn.Sequential(*layers)
            if sigma:
                self.nn = SigmaNet(
                    self.nn,
                    sigma=Lip,
                    monotone_constraints=monotone_constraints[important_feature_idxs],
                )


        def forward(self, x):
            return self.nn(x)

    model = Model(robust=True, sigma=monotonic).to(device)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    EPOCHS = 1000
    print("params:", sum(p.numel() for p in model.parameters()))
    bar = tqdm(range(EPOCHS))
    best_rmse = 1
    for epoch in bar:
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
                f"train rmse: {trrmse:.4f} test rmse: {tsrmse:.4f}, best: {best_rmse:.4f}, lr: {optimizer.param_groups[0]['lr']:.4f}"
            )
    return best_rmse

rmses = [run_exp(max_lr=2e-4, expwidth=3, depth=3, batchsize=2**8, seed=i, Lip=1, monotonic=True) for i in range(3)]
print(f"mean: {np.mean(rmses):.4f}, std: {np.std(rmses):.4f}")

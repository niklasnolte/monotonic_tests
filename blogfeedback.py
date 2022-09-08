# %%
import torch
import lightgbm as lgb
from tqdm import tqdm
from loaders.blog_loader import load_data, mono_list
from monotonenorm import SigmaNet, GroupSort
import optuna

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# %%
Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False)
input_dim = Xtr.shape[1]

# %%
monotone_constraints = [1 if i in mono_list else 0 for i in range(input_dim)]

# %%
# clf = lgb.LGBMRegressor(
#     n_estimators=10000,
#     max_depth=5,
#     learning_rate=0.1,
#     monotone_constraints=monotone_constraints,
# )
# clf.fit(
#     Xtr,
#     Ytr,
#     early_stopping_rounds=200,
#     eval_set=[(Xts, Yts)],
#     eval_metric="rmse",
#     verbose=0,
# )

# # %%
# rmse_tr = (((clf.predict(Xtr) - Ytr) ** 2).mean()) ** 0.5
# rmse_ts = (((clf.predict(Xts) - Yts) ** 2).mean()) ** 0.5
# print(rmse_tr, rmse_ts)


# %%


def run_exp(
    max_lr=2e-3, expwidth=2, depth=4, Lip=8, monotonic=True, batchsize=256, seed=1
):
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
                    torch.nn.Linear(input_dim, width),
                    kind="one-inf",
                    alpha=per_layer_lip,
                ),
                activation(),
            ]
            for _ in range(depth - 2):
                layers.append(
                    direct_norm(
                        torch.nn.Linear(width, width), kind="inf", alpha=per_layer_lip
                    )
                )
                layers.append(activation())

            layers.append(
                direct_norm(torch.nn.Linear(width, 1), kind="inf", alpha=per_layer_lip)
            )

            self.nn = torch.nn.Sequential(*layers)
            if sigma:
                self.nn = SigmaNet(
                    self.nn,
                    sigma=per_layer_lip ** depth,
                    monotone_constraints=monotone_constraints,
                )

            if monotonic:
                # increase std of weights accordingly
                for m in self.nn.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.orthogonal_(m.weight)

        def forward(self, x):
            return self.nn(x)

    model = Model(robust=True, sigma=monotonic).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    EPOCHS = 2000
    lr_final = 1e-4
    gamma = (lr_final / max_lr) ** (1 / (EPOCHS * len(dataloader)))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    print("params:", sum(p.numel() for p in model.parameters()))
    bar = tqdm(range(EPOCHS))
    best_rmse = 1
    for _ in bar:
        for Xi, yi in dataloader:
            y_pred = model(Xi)
            losstr = torch.nn.functional.mse_loss(y_pred, yi)
            optimizer.zero_grad()
            losstr.backward()
            optimizer.step()
            # scheduler.step()
        with torch.no_grad():
            y_predts = model(Xtst)
            lossts = torch.nn.functional.mse_loss(y_predts, Ytst)
            tsrmse = lossts.item() ** 0.5
            trrmse = losstr.item() ** 0.5
            best_rmse = min(best_rmse, tsrmse)
            bar.set_description(
                f"train rmse: {trrmse:.4f} test rmse: {tsrmse:.4f}, best: {best_rmse:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}"
            )
    return best_rmse


# run_exp(max_lr=7e-4, expwidth=2, depth=4, batchsize=1024, monotonic=True, seed=10, Lip=.5)
# exit()

# %%
# run with optuna
def objective(trial):
    # balance on two gpus

    max_lr = trial.suggest_loguniform("max_lr", 5e-4, 3e-3)
    width = trial.suggest_int("expwidth", 1, 4)
    depth = trial.suggest_int("depth", 3,5)
    batchsize = trial.suggest_int("batchsize", 128, 1024)
    # seed = trial.suggest_int("seed", 1, 10)
    return run_exp(
        max_lr=max_lr, expwidth=width, batchsize=batchsize, seed=4, depth=depth, Lip=20
    )


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, n_jobs=8)

# %%
# print results
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
print("  Value: {}".format(study.best_trial.value))
# %%
# save
import pickle

with open("optuna_blogfeedback.pkl", "wb") as f:
    pickle.dump(study, f)


# monotonic network
# f(x) = g(x) + 1*x , g(x) \in L^k, k = 10

# x_i ~ P_x(mu=0,std=1), iid
# g(x) ~ P_g(mu=0,std=O(1)), assume E(W_i) = 0, E(W_i^2) = 5 / fan_in

# want f(x) ~ P_f(mu=0,std=1)


# norm preserving maps:
# <Ax, Ay> = <x, y

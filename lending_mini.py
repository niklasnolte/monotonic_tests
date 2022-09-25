import torch
from tqdm import tqdm
from loaders.lending_loader import load_data, mono_list
from monotonenorm import SigmaNet, GroupSort
from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm as lgb

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False)
monotonic_constraints = np.array([int(i in mono_list) for i in range(Xtr.shape[1])])

#lightgbm
clf = lgb.LGBMClassifier(
    n_estimators=10000,
    max_depth=5,
    learning_rate=0.1,
    monotone_constraints=monotonic_constraints,
)

clf.fit(
    Xtr,
    Ytr,
    early_stopping_rounds=200,
    eval_set=[(Xts, Yts)],
    eval_metric="acc",
    verbose=0,
)

#print accuracy
print('lightgbm', accuracy_score(Yts, clf.predict(Xts)))

# only take top 5 features:
top_features = np.array(list(set(mono_list) | set(np.argsort(clf.feature_importances_)[-10:])))
Xtr = Xtr[:, top_features]
Xts = Xts[:, top_features]

def run_exp(seed):
  torch.manual_seed(seed)

  per_layer_lip = 1
  width = 6

  class Model(torch.nn.Module):
    def __init__(self, width, robust=False, sigma=False):
      super().__init__()
      if robust:
        from monotonenorm import direct_norm
        activation = lambda : GroupSort(width//2)
      else:
        direct_norm = lambda x, *args, **kwargs: x # make it a normal network
        activation = lambda : torch.nn.ReLU()

      self.nn = torch.nn.Sequential(
        direct_norm(torch.nn.Linear(Xtr.shape[1], width), kind="one-inf", max_norm=per_layer_lip),
        activation(),
        direct_norm(torch.nn.Linear(width, width), kind="inf", max_norm=per_layer_lip),
        # activation(),
        # direct_norm(torch.nn.Linear(width, width), kind="inf", max_norm=per_layer_lip),
        activation(),
        direct_norm(torch.nn.Linear(width, 1), kind="inf", max_norm=per_layer_lip),
      )
      if sigma:
        self.nn = torch.nn.Sequential(
          SigmaNet(self.nn, sigma=per_layer_lip**4, monotone_constraints=monotonic_constraints[top_features]),
          torch.nn.Sigmoid()
        )
    
    def forward(self, x):
      return self.nn(x)

  model = Model(width, robust=True, sigma=True)



  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  print('params:', sum(p.numel() for p in model.parameters()))
  print(model)

  Xtrt = torch.tensor(Xtr, dtype=torch.float32).to(device)
  Ytrt = torch.tensor(Ytr, dtype=torch.float32).view(-1, 1).to(device)
  Xtst = torch.tensor(Xts, dtype=torch.float32).to(device)
  Ytst = torch.tensor(Yts, dtype=torch.float32).view(-1, 1).to(device)

  mean = Xtrt.mean(0)
  std = Xtrt.std(0)
  Xtrt = (Xtrt - mean) / std
  Xtst = (Xtst - mean) / std


  dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtrt, Ytrt), batch_size=int(2**9), shuffle=True)
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
      if i % 1 == 0:
        acc = 0
        for i in np.linspace(0, 1, 50):
          acc = max(acc, accuracy_score(Ytst.cpu().numpy(), y_predts.cpu().numpy()>i))

      max_acc = max(max_acc, acc)
      bar.set_description(f'Loss: {losstr.item():.4f} {lossts.item():.4f}, acc: {acc.item():.4f}, max_acc: {max_acc:.4f}')
  return max_acc

if __name__ == "__main__":
  accs = [run_exp(i) for i in range(3)] # 3 seeds
  #print mean and std of the 3 runs
  print(f"mean: {np.mean(accs):.4f}, std: {np.std(accs):.4f}")

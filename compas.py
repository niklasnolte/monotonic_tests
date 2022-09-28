import torch
import torch.nn as nn
from loaders.compas_loader import load_data
import torch.utils.data as Data
from monotonenorm import SigmaNet, direct_norm, GroupSort
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, y_train, X_test, y_test = load_data(get_categorical_info=False)

feature_num = X_train.shape[1]
mono_feature = 4
X_train = torch.tensor(X_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_train = torch.tensor(y_train).float().unsqueeze(1).to(device)
y_test = torch.tensor(y_test).float().unsqueeze(1).to(device)
data_train = Data.TensorDataset(X_train, y_train)

print(X_train.shape)

criterion = nn.BCEWithLogitsLoss()

per_layer_lip = 4


def run(seed):
  acc = 0
  torch.manual_seed(seed)

  width = 2
  
  network = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(13, width), kind="one-inf", max_norm=per_layer_lip),
    GroupSort(width//2),
    direct_norm(torch.nn.Linear(width, width), kind="inf", max_norm=per_layer_lip),
    GroupSort(width//2),
    direct_norm(torch.nn.Linear(width, width), kind="inf", max_norm=per_layer_lip),
    GroupSort(width//2),
    direct_norm(torch.nn.Linear(width, 1), kind="inf", max_norm=per_layer_lip),
  )
  network = SigmaNet(network, sigma=per_layer_lip**4, monotone_constraints=[1]*mono_feature + [0]*(13-mono_feature))
  
  network = network.to(device) 

  nparams = sum(p.numel() for p in network.parameters() if p.requires_grad)
  print('total param amount:', nparams)

  optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
  
  data_train_loader = Data.DataLoader(
      dataset=data_train,      
      batch_size=256,
      shuffle=True,
  )
  bar = tqdm(range(1000))
  for i in bar:
    for X,y in data_train_loader:
      y_pred = network(X)
      loss_train = criterion(y_pred, y)
      optimizer.zero_grad()
      loss_train.backward()
      optimizer.step()
    
    #test
    y_pred = network(X_test)
    loss = criterion(y_pred, y_test)
    # accuracy
    acci = 0
    for i in torch.linspace(0,1,50):
      acci = max(acci, accuracy_score(y_test.cpu().detach().numpy(), (y_pred.cpu().detach().numpy() > i.item()).astype(int)))
    
    acc = max(acc, acci)
    bar.set_description(f"train: {loss_train.item():.4f}, test: {loss.item():.4f}, current acc: {acci:.4f}, best acc: {acc:.4f}")
  return acc  

accs = [run(i) for i in range(3)]
print(f"mean: {np.mean(accs)}, std: {np.std(accs)}")

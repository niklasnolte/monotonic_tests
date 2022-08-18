### simple 2-d case with PyTorch
### monotonic: capital-gain, weekly hours of work and education level, and the gender wage gap
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from compas_loader import load_data
import torch.utils.data as Data
from monotonenorm import SigmaNet, direct_norm, GroupSort
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
from sys import argv

X_train, y_train, X_test, y_test, start_index, cat_length = load_data(get_categorical_info=True)
n = X_train.shape[0]
n = int(0.8*n)
X_val = X_train[n:, :]
y_val = y_train[n:]
X_train = X_train[:n, :]
y_train = y_train[:n]

feature_num = X_train.shape[1]
mono_feature = 4
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
X_val = torch.tensor(X_val).float()
y_train = torch.tensor(y_train).float().unsqueeze(1)
y_test = torch.tensor(y_test).float().unsqueeze(1)
y_val = torch.tensor(y_val).float().unsqueeze(1)
data_train = Data.TensorDataset(X_train, y_train)

data_train_loader = Data.DataLoader(
    dataset=data_train,      
    batch_size=512,     
    shuffle=True,               
    num_workers=2,
)

print(X_train.shape)

criterion = nn.BCEWithLogitsLoss()

if "lip" in argv:
  per_layer_lip = 3

  network = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(13, 100), kind="one-inf", alpha=per_layer_lip),
    GroupSort(2),
    direct_norm(torch.nn.Linear(100, 100), kind="inf", alpha=per_layer_lip),
    GroupSort(2),
    direct_norm(torch.nn.Linear(100, 100), kind="inf", alpha=per_layer_lip),
    GroupSort(2),
    direct_norm(torch.nn.Linear(100, 1), kind="inf", alpha=per_layer_lip),
  )
  network = SigmaNet(network, sigma=per_layer_lip**4, monotone_constraints=[1]*mono_feature + [0]*(13-mono_feature))
else:
  network = torch.nn.Sequential(
    torch.nn.Linear(13, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1),
  )
  

#net = MLP_relu(mono_feature=mono_feature, non_mono_feature=feature_num-mono_feature, mono_sub_num=1, non_mono_sub_num=1, mono_hidden_num = 256, non_mono_hidden_num=100)
param_amount = 0
for p in network.named_parameters():
    print(p[0], p[1].numel())
    param_amount += p[1].numel()
print('total param amount:', param_amount)

net = network.cuda()
X_test = X_test.cuda()
y_test = y_test.cuda()
X_val = X_val.cuda()
y_val = y_val.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)

# train loop
bar = tqdm(range(1000))

for i in bar:
  for X,y in data_train_loader:
    X = X.cuda()
    y = y.cuda()
    y_pred = network(X)
    loss_train = criterion(y_pred, y)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    scheduler.step()
  
  #test
  y_pred = network(X_test)
  loss = criterion(y_pred, y_test)
  # accuracy
  bacc = 0
  acc = 0
  for i in torch.linspace(0,1,50):
    bacc = max(bacc, balanced_accuracy_score(y_test.cpu().detach().numpy(), (y_pred.cpu().detach().numpy() > i.item()).astype(int)))
    acc = max(acc, accuracy_score(y_test.cpu().detach().numpy(), (y_pred.cpu().detach().numpy() > i.item()).astype(int)))
  
  bar.set_description(f"train: {loss_train.item():.4f}, test: {loss.item():.4f}, bacc: {bacc:.4f}, acc: {acc:.4f}")

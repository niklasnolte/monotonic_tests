# %%
import torch
from tqdm import tqdm

# %%
basepath = '/data/nnolte/chest_xray/'
XIMG = torch.load(basepath+"XIMG.pt")

# make a dataloader
from torch.utils.data import DataLoader
from torchvision import transforms

loader = DataLoader(XIMG, batch_size=32, shuffle=False)

# %%
device = torch.device("cpu")#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval().to(device)
resnet.fc = torch.nn.Identity()

# %%
with torch.no_grad():
  new_features = [resnet(x) for x in tqdm(loader)]

# %%
new_features = torch.cat(new_features).float()

# %%
torch.save(new_features, basepath+"resnet18_features.pt")

# %%




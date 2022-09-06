# %%
import torch
from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# %%
basepath = '/data/nnolte/chest_xray/'
XIMG = torch.load(basepath+"XIMG.pt")
# %%

loader = DataLoader(XIMG, batch_size=128, shuffle=False)

# %%
device = torch.device("cpu")#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()

# %%
with torch.no_grad():
  new_features = [resnet(x) for x in tqdm(loader)]

# %%
new_features = torch.cat(new_features).float()

# %%
torch.save(new_features, basepath+"resnet18_features.pt")


import os
import torch
from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.data import DataLoader

from chest_config import basepath


XIMG = torch.load(os.path.join(basepath, "XIMG.pt"))

loader = DataLoader(XIMG, batch_size=128, shuffle=False)

resnet = resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()

with torch.no_grad():
  new_features = [resnet(x) for x in tqdm(loader)]

new_features = torch.cat(new_features).float()

torch.save(new_features, os.path.join(basepath, "resnet18_features.pt"))


# %%
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import os
import PIL

# %%
basepath = "/data/nnolte/chest_xray/"

# %%
sample = pd.read_csv(basepath+"sample_labels.csv")

# %%
#read images with PIL
full_sized = [PIL.Image.open(os.path.join(basepath+'images', imgidx)).convert("RGB") for imgidx in sample["Image Index"]]

# %%
Y = sample["Finding Labels"] != "No Finding"

# according to https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
imagenet_dim = 224
#imagenet transformations
transforms_ = transforms.Compose([
    transforms.Resize(imagenet_dim),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

XIMG = torch.stack([transforms_(x) for x in full_sized])
# %%
follow_up_num = sample["Follow-up #"]
age = sample["Patient Age"].apply(lambda x: int(x[:-1]))
gender = sample["Patient Gender"] == "M"
vp = sample["View Position"] == "PA"
XTAB = np.hstack([follow_up_num.to_numpy()[:,None],
                  age.to_numpy()[:,None],
                  gender.to_numpy()[:,None],
                  vp.to_numpy()[:,None]])
XTAB = torch.from_numpy(XTAB).float()

# %%
torch.save(XIMG, basepath+"XIMG.pt")
torch.save(XTAB, basepath+"XTAB.pt")
torch.save(Y, basepath+"Y.pt")



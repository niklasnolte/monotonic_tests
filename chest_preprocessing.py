# %%
import pandas as pd
import numpy as np
import torch
import os
import cv2

# %%
basepath = "/data/nnolte/chest_xray/"

# %%
sample = pd.read_csv(basepath+"sample_labels.csv")

# %%
full_sized = [cv2.imread(os.path.join(basepath+'images', imgidx)) for imgidx in sample["Image Index"]]


# %%
imagenet_dim = 224
Y = sample["Finding Labels"] != "No Finding"
XIMG = np.array([cv2.resize(img, (imagenet_dim, imagenet_dim), interpolation=cv2.INTER_CUBIC) for img in full_sized])
XIMG = torch.from_numpy(XIMG).float().permute(0, 3, 1, 2)

# %%
follow_up_num = sample["Follow-up #"]
age = sample["Patient Age"].apply(lambda x: int(x[:-1]))
XTAB = np.hstack([follow_up_num.to_numpy()[:,None], age.to_numpy()[:,None]])
XTAB = torch.from_numpy(XTAB).float()

# %%
torch.save(XIMG, basepath+"XIMG.pt")
torch.save(XTAB, basepath+"XTAB.pt")
torch.save(Y, basepath+"Y.pt")



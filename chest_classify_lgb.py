# %%
import lightgbm as lgb
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# %%
# load data
basepath = "/data/nnolte/chest_xray/"
XTAB = torch.load(basepath + "XTAB.pt")
Y = torch.load(basepath + "Y.pt").to_numpy()
resnet_features = torch.load(basepath + "resnet18_features.pt")

# %%
# X = XTAB.numpy()
X = torch.hstack([XTAB, resnet_features]).numpy()

# %%
accs = []
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=i
    )
    # %%
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=i,
        max_depth=10,
        num_leaves=50,
        monotone_constraint=[1,1,0,0] + [0]*resnet_features.shape[1],
    )
    clf.fit(
        verbose=False,
        X=X_train,
        y=y_train,
        early_stopping_rounds=100,
        eval_set=[(X_test, y_test)],
        eval_metric='None'
    )
    # %%
    acc = 0
    for i in range(0, 1, 100):
        acc = max(acc, accuracy_score(y_test, clf.predict(X_test) > i))

    bacc = 0
    for i in range(0, 1, 100):
        bacc = max(bacc, balanced_accuracy_score(y_test, clf.predict(X_test) > i))
    # %%
    print(f"accuracy: {acc:.5f}, balanced accuracy: {bacc:.5f}")
    accs.append(acc)

print(f"mean accuracy: {np.mean(accs):.5f}, std accuracy: {np.std(accs):.5f}")
# %%

import numpy as np
import scipy.stats as stats
import pandas as pd
from resample import permutation, bootstrap, utils
from sklearn.datasets import load_boston
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict


def fitgbm(A):
    reg = LinearRegression()
    X = A[:, :A.shape[1]-1]
    y = A[:, A.shape[1]-1]
    reg.fit(X, y)
    print(type(reg))
    return reg


bost = load_boston()
df = pd.DataFrame(bost["data"], columns=bost["feature_names"])
y = pd.Series(np.log(bost["target"]), name="target")
boot_mod = bootstrap.bootstrap(a=df.join(y).values, f=fitgbm, b=500)
for m in boot_mod:
    preds = m.predict(df.values)

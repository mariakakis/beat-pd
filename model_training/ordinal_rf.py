from settings import *
from sklearn.base import clone, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


# https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
class OrdinalRandomForestClassifier(RandomForestClassifier, ClassifierMixin):
    def __init__(self, **kwargs):
        RandomForestClassifier.__init__(self, kwargs)
        self.base_clf = RandomForestClassifier()
        self.clfs = {}
        self.unique_class = None

    def fit(self, x, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] >= 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary model_training problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.base_clf)
                clf.fit(x, binary_y)
                self.clfs[i] = clf
        return self

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x):
        clfs_predict = {k: self.clfs[k].predict_proba(x) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[y - 1][:, 1] - clfs_predict[y][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y - 1][:, 1])
        return np.vstack(predicted).T

    # def get_params(self, deep=True):
    #     return self.base_clf.get_params(deep)

    def set_params(self, **params):
        self.base_clf.set_params(**params)
        for clf in self.clfs:
            clf.set_params(**params)
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

from sklearn.base import BaseEstimator, TransformerMixin

class EmbeddingPassthrough(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)
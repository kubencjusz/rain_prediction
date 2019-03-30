from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion
import pandas as pd
import numpy as np

# create a feature - probability of rain given categorical feature level
class ProbRain(TransformerMixin):
    def __init__(self, features=["Location"]):
        self.feat = features
        self.prob = []

    def fit(self, X, y, **fit_params):
        assert X.shape[0]==y.shape[0], "Number of observations in X and y don't match!"
        for feat in self.feat:
            tmp = pd.concat([X[feat], pd.DataFrame(y, index=X.index, columns=["RainTomorrow"])], axis=1)
            tmp = tmp.groupby(feat)['RainTomorrow'].value_counts().unstack(1)
            tmp = tmp.apply(lambda x: x/tmp.sum(1)).stack().to_frame(str(feat)+"Prob").reset_index()
            self.prob.append(tmp.query("RainTomorrow==1").drop(columns="RainTomorrow"))
        return self

    def transform(self, X):
        for df in self.prob:
            X = pd.merge(X, df, how="left", on=df.columns.values[0])
        return X

# it works:
#ProbRain(features=["Location", "WindGustDir"]).fit_transform(X, y)

# assign a categorical feature's level its count
class CounterCat(TransformerMixin):
    def __init__(self, cols=["Location"]):
        self.counts = {}
        self.columns = cols

    def fit(self, X, y=None, **fit_params):
        assert any([x in X.columns.values for x in self.columns]), "There are no columns to be transformed!"
        for col in self.columns:
            self.counts[col] = X[col].value_counts()
        return self

    def transform(self, X):
        # if level not found in dictionary assign -1
        tmp = pd.concat([X[col].apply(lambda x: self.counts[col].get(x, -1))
                         for col in self.columns],
                        axis=1)
        tmp.columns = [col+"Count" for col in tmp.columns]
        return pd.concat([X, tmp], axis=1)

# it works:
#CounterCat().fit_transform(X)

# imputer of NaN
class Imputer(TransformerMixin):
    def __init__(self, strategy="median"):
        self.strat = strategy
        self.col_values = {}
        self.fun_dict = {"median": np.nanmedian,"mean": np.nanmean,
                         "most_frequent": lambda x: x.value_counts().argsort().index[0],
                         }

    def fit(self, X, y=None):
        for col in X.columns:
            self.col_values[col] = self.fun_dict[self.strat](X[col])
        return self

    def transform(self, X):
        return X.fillna(value=self.col_values)


# standard scaler
class Standardize(TransformerMixin):
    def __init__(self):
        self.mean = {}
        self.std = {}

    def fit(self, X, y=None):
        assert all(X.dtypes != "object"), "Standarization can be made only on numeric features"
        for col in X.columns:
            self.mean[col] = np.nanmean(X[col])
            self.std[col] = np.nanstd(X[col])
        return self

    def transform(self, X, y=None):
        num_features = ~ (X.dtypes == "object")
        num_features = X.columns.values[num_features]

        for col in num_features:
            X[col] -= self.mean[col]
            X[col] /= self.std[col]
        return X

class GetDateParts(TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Month'] = X['Date'].str[5:7]
        X['Day'] = X['Date'].str[8:10]
        X['Year'] = X['Date'].str[:4]
        return X

class OneHotEncode(TransformerMixin):
    def __init__(self, exclude = ["Date"]):
        self.exclude = exclude
        self.cat_dict = {}
        self.possible_cats = []

    def which_features(self, X):
        cat_feat = X.dtypes == "object"
        cat_feat = X.columns.values[cat_feat].tolist()
        for f in self.exclude:
            try:
                cat_feat.remove(f)
            except ValueError:
                pass
        return cat_feat

    def fit(self, X, y=None):
        cat_feat = self.which_features(X)
        for f in cat_feat:
            self.cat_dict[f] = X[f].unique()


        for key, value in self.cat_dict.items():
            for val in value:
                self.possible_cats.append(key+"_"+str(val))
        return self

    def transform(self, X):
        cat_feat = self.which_features(X)

        X = pd.get_dummies(X[cat_feat])
        # remove columns that may be new in test data
        X.reindex(columns=self.possible_cats, fill_value=0)

        return X

# restore pandas dataframes
def _transform_one(transformer, X, y, weight, **fit_params):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(transformer, X, y, weight, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transform

class FeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y, **fit_params)
            for name, trans, weight in self._iter())

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        return pd.concat(Xs, axis=1)

    def transform(self, X, **kwargs):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X, **kwargs)
            for name, trans, weight in self._iter())
        return pd.concat(Xs, axis=1)
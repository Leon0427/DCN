import pandas as pd


class FeatureDictionary(object):
    def __init__(self, train_file=None, test_file=None, df_train=None, df_test=None, numeric_cols=[], ignored_cols=[]):
        self.feature_dim = None
        self.feature_dict = {}
        assert not ((train_file is None) and (df_train is None))
        assert not ((train_file is not None) and (df_train is not None))
        assert not ((test_file is None) and (df_test is None))
        assert not ((test_file is not None) and (df_test is not None))
        self.train_file = train_file
        self.test_file = test_file
        self.df_train = df_train
        self.df_test = df_test
        if self.df_train is None:
            self.df_train = pd.read_csv(self.train_file)
        if self.df_test is None:
            self.df_test = pd.read_csv(self.test_file)
        self.numeric_cols = numeric_cols
        self.ignored_cols = ignored_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        df_train = self.df_train
        df_test = self.df_test
        df = pd.concat([df_train, df_test])
        feature_dim = 0
        for col in df.columns:
            if col in self.ignored_cols:
                continue
            if col in self.numeric_cols:
                self.feature_dict[col] = feature_dim
                feature_dim += 1
            else:
                unique_categories = df[col].unique()
                self.feature_dict[col] = dict(
                    zip(unique_categories, range(feature_dim, len(unique_categories) + feature_dim)))
                feature_dim += len(unique_categories)
        self.feature_dim = feature_dim


class DataParser(object):
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict

    def parse(self, in_file=None, df=None, has_label=False):
        assert not ((in_file is None) and (df is None))
        assert not ((in_file is not None) and (df is not None))
        if in_file is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(in_file)
        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            dfi.drop(["id"], axis=1, inplace=True)

        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feature_dict.ignored_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feature_dict.numeric_cols:
                dfi[col] = self.feature_dict.feature_dict[col]
            else:
                # series.map
                dfi[col] = dfi[col].map(self.feature_dict.feature_dict[col])
                dfv[col] = 1.

        Xi = dfi.values.tolist()
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=['CAEC', 'CALC']):
        self.feature_to_drop = feature_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.feature_to_drop)


class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, cols=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']):
        self.cols = cols
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = self.scaler.transform(X[self.cols])
        return X


class OnHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['MTRANS']):
        self.OneHotEncoding = OneHotEncoding
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )

    def fit(self, X, y=None):
        self.encoder.fit(X[self.OneHotEncoding])
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X[self.OneHotEncoding])
        feature_names = self.encoder.get_feature_names_out(self.OneHotEncoding)

        df_encoded = pd.DataFrame(
            encoded,
            columns=feature_names,
            index=X.index
        )

        df_rest = X.drop(columns=self.OneHotEncoding)
        return pd.concat([df_rest, df_encoded], axis=1)

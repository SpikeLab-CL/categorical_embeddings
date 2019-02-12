from base_model import BaseModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder
import numpy as np

debug = True
if(debug):
    import pandas as pd

class Embedder(BaseModel):
    model = None
    def __init__(self, target_type=None):
        BaseModel.__init__(self, target_type=target_type)
    
    def _prepare_feature(self, X):
        if X.dtypes == "object":
            encoder = LabelEncoder()
            X = encoder.fit_transform(X)
            return X, encoder
        else:
            raise RuntimeError("<Series> must be of type 'object'")
    
    def _prepare_target(self, y, target_type):
        try:
            if target_type == "regression":
                if  (y.dtypes != "int64" and y.dtypes != "float64"):
                    raise RuntimeError("For target_type='regression' target must be <int64> or <float64>")
                else:
                    return y
            elif target_type == "binary_classification":
                assert y.nunique() == 2, "For target_type='binary_classification' target must have only 2 classes"
                if (y.dtypes == "bool" or y.dtypes == "object"):
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(y)
                    return y
                else:
                    raise RuntimeError("For target_type='binary_classification' target must be <bool> or <object>")
            else:
                encoder = OneHotEncoder()
                y = encoder.fit_transform(y.values.reshape(1, -1))
                return y
        except Exception as e:
            print("Error:", e)

    def _get_model_params(self, X):
            return np.unique(X).shape[0], min(np.ceil((np.unique(X).shape[0])/2),50)

    def fit(self,X, y):
        """
        X: pandas.core.series.Series: Series with the data
        y: pandas.core.series.Series: Series with target variable
        return pd.DataFrame() with encodings values
        """
        data = X.copy(deep=True)
        target = y.copy(deep=True)
        data, encoder = self._prepare_feature(data)
        y = self._prepare_target(y=y,target_type=self.target_type)
        n_classes, embedding_size = self._get_model_params(data)
        self.model = self.build_model(num_classes=n_classes, vector_size=int(embedding_size))
        self.model.fit(x=data, y=y, epochs=30)

if __name__ == "__main__":
    data = pd.read_csv("/Users/maravenag/Desktop/categorical_embedding/Pokemon.csv")
    data.fillna("NaN",inplace=True)
    e = Embedder(target_type="regression")
    e.fit(data['Type 1'], data['Total'])
from categorical_embedding.base_model import BaseModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

class Embedder(BaseModel):
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
        if target_type == "regression":
            if  (y.dtypes != "int64" and y.dtypes != "float64"):
                raise RuntimeError("For target_type='regression' target must be <int64> or <float64>")
            else:
                scaler = MinMaxScaler()
                return scaler.fit_transform(y.values.reshape(-1, 1))
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

    def _get_model_params(self, X):
            return np.unique(X).shape[0], min(np.ceil((np.unique(X).shape[0])/2),50)

    def _get_components(self, embedding_size, feature_name):
        col_names = [ feature_name+"_{0}".format(x) for x in range(1,int(embedding_size)+1)]
        embedding_layer = self.model.get_layer(name="embedding_layer")
        embedding_layer = pd.DataFrame(embedding_layer.get_weights()[0])
        embedding_layer.columns = col_names
        return embedding_layer

    def _parse_components_index(self, components, categorical_names, feature_name):
        indexs = pd.Series(categorical_names).to_frame()
        indexs.columns = [feature_name]
        return pd.merge(components,indexs, how="left", left_index=True, right_index=True)

    def fit(self,X, y):
        """
        X: pandas.core.series.Series: Series with the data
        y: pandas.core.series.Series: Series with target variable
        return pd.DataFrame() with encodings values
        """
        data = X.copy(deep=True)
        feature_name = data.name.replace(" ","_")
        target = y.copy(deep=True)
        data, encoder = self._prepare_feature(data)
        y = self._prepare_target(y=y,target_type=self.target_type)
        n_classes, embedding_size = self._get_model_params(data)
        categorical_names = list(encoder.inverse_transform([x for x in range(0,n_classes)]))
        self.model = self.build_model(num_classes=n_classes, vector_size=int(embedding_size))
        self.fit_model(X=data, y=y)
        components = self._get_components(embedding_size, feature_name)
        components = self._parse_components_index(components, categorical_names, feature_name)
        return components
    
    def fit_transform(self, X, y, exclude_columns=[]):
        """
        X: pandas.DataFrame(): Series with the data
        y: <string> target variable
        return pd.DataFrame() with encodings values
        """
        data = X.copy(deep=True)
        selected_cols = data.select_dtypes(['object'])
        y = data[y]
        items = [x for x in selected_cols.columns if x not in exclude_columns]
        selected_cols = selected_cols.filter(items=items)
        pbar = tqdm(total=len(items))
        for column in selected_cols:
            component = self.fit(selected_cols[column], y=y)
            data = pd.merge(data, component, how="left")
            pbar.update(1)
        pbar.close()
        return data

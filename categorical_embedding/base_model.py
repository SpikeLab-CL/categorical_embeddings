from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.callbacks import EarlyStopping

class BaseModel():
    model = None
    def __init__(self, target_type=None):
        assert target_type in ['regression',
                               'binary_classification',
                               'multiclass'],"target_type must be 'regression' or 'binary_classification' or 'multiclass'"
        self.target_type = target_type

    def build_model(self, num_classes=2, vector_size=2):
        model = Sequential()
        model.add(Embedding(input_dim=num_classes,
                            output_dim=vector_size,
                            input_length=1,
                            name="embedding_layer"))
        model.add(Flatten())
        model.add(Dense(int(1.5*num_classes), activation="relu"))
        model.add(Dense(int(0.5*num_classes), activation="relu"))
        if self.target_type == "regression":
            model.add(Dense(1, activation="linear"))
            model.compile(loss='mse', optimizer='adam')
        elif self.target_type == "binary_classification":
            model.add(Dense(1, activation="sigmoid"))
            model.compile(loss='binary_crossentropy', optimizer='adam')
        else:
            model.add(Dense(num_classes, activation="softmax"))
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
    def fit_model(self, X, y):
        stopping = EarlyStopping(monitor='loss', patience=15)
        self.model.fit(x=X, y=y, epochs=100, verbose=0, callbacks=[stopping])
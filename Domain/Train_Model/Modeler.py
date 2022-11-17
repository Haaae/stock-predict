from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError, Accuracy

def get_model(loss_function, optimizer, patience, seq_len):
    model = My_LSTM(seq_len)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=[RootMeanSquaredError(), Accuracy()])
    early_stopping = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)
    # verbose = early_stopping이 적용됐다고 표시
    # restore_best_weights=True : early_stopping 시 가장 val_loss가 낮았던 때로 모델 롤백
    
    return model, early_stopping

def My_LSTM(seq_len):    
    input1 = Input(shape=(seq_len, 5))
    dense1 = LSTM(256)(input1)
    dense1 = Dense(64, activation='relu')(dense1)
    dense1 = Dense(32, activation='relu')(dense1)
    dense1 = Dense(32)(dense1)
    dense1 = Dense(32)(dense1)
    dense1 = Dense(32)(dense1)
    output1 = Dense(1)(dense1)

    model = Model(inputs=input1, outputs=output1)

    return model

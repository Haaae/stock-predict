import os

from Domain.Train_Model.Modeler import My_LSTM

def load_model(model_save_path, stock, seq_len) :

    dir_list = os.listdir(model_save_path)

    if not stock + '.index' in dir_list:
        raise Exception("저장된 모델이 없습니다.")

    print(stock + "_Model exist.")

    model = My_LSTM(seq_len)
    model.load_weights(model_save_path + '\\' + stock)

    return model

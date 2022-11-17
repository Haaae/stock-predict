import os

from src.Domain.Train_Model.Modeler import My_LSTM

def load_model(stock, seq_len) :

    dir_path = "C:\\Users\\82106\\Desktop\\Stock\\Predict\\Model"
    model_path = dir_path + "\\" +  stock
    dir_list = os.listdir(dir_path)

    if not stock + '.index' in dir_list:
        raise Exception("저장된 모델이 없습니다.")

    print(stock + "_Model exist.")

    model = My_LSTM(seq_len)
    model.load_weights(model_path)

    return model

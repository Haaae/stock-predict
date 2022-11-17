from datetime import datetime
import warnings
import tensorflow as tf

from Train_Model.Data_Loader import load_data
from Train_Model.Tester import test_model
from Train_Model.Trainer import model_train
from Train_Model.Stock_Loader import load_stocks

def main():
    warnings.filterwarnings("ignore")

    # Hyper Parameter
    date_time = datetime.now().strftime("%Y_%m_%d")   # 학습 시작 시간
    start = datetime(2015, 1, 1)    # 주식 기간 2015
    end = datetime.today()     # 주식 기간
    seq_len = 5
    test_rate = 0.2
    loss_function = 'mse'
    optimizer = 'adam'
    patience = 20
    batch_size = 8
    epoch = 50
    validation_split = 0
    stocks_path = "C:\\Users\\82106\\Desktop\\Stock\\Predict\\Stocks.txt"

    # 텐서 보드
    logdir="TensroBoard_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


    # 2022-11-16 기준
    stocks = load_stocks(stocks_path)

    for stock in stocks :
        print('\n\n')
        print("*** " + stock + " Train Start ***")
        stock_ticker = stocks[stock]
        model_save_path = "C:\\Users\\82106\\Desktop\\Stock\\Predict\\Model\\" + stock

        # Load dataset
        x_train, x_test, y_train, y_test, x_tomorrow, x_close = load_data(stock_ticker, start, end, seq_len, test_rate)

        # Get trained model
        model = model_train(loss_function,
                                optimizer,
                                patience,
                                seq_len,
                                x_train,
                                y_train,
                                batch_size,
                                epoch,
                                validation_split,
                                tensorboard_callback)

        # model.summary()

        test_model(model, x_test, y_test, stock)

        model.save_weights(model_save_path)

if __name__ == "__main__":
    main()

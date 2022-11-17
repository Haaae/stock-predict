from datetime import datetime
import warnings

from Predict_Stock.Predictor import predict_tomorrow
from Predict_Stock.Loader import load_model
from Domain.Train_Model.Data_Loader import load_data
from Domain.Train_Model.Tester import test_model
from Domain.Train_Model.StockList_Loader import load_stocks

def main():
    warnings.filterwarnings("ignore")

    # Hyper Parameter
    date_time = datetime.now().strftime("%Y_%m_%d")   # 학습 시작 시간
    start = datetime(2015, 1, 1)    # 주식 기간 2015
    end = datetime.today()     # 주식 기간
    seq_len = 5
    stocks_path_10per = ".\\stocks\\testRate_10per.txt"
    stocks_path_20per = ".\\stocks\\testRate_20per.txt"
    model_save_path = ".\\Model"

    stocksPath_testRate = {stocks_path_10per:0.1, stocks_path_20per:0.2}

    for stocks_path in stocksPath_testRate :

        test_rate = stocksPath_testRate[stocks_path]

        # 2022-11-16 기준
        stocks = load_stocks(stocks_path)

        print("test rate : ", test_rate)

        for stock in stocks:
            print('\n\n')
            print("*** " + stock + " Predict Start ***")
            stock_ticker = stocks[stock]

            # Load dataset
            x_train, x_test, y_train, y_test, x_tomorrow, x_close = load_data(stock_ticker, start, end, seq_len, test_rate)

            _predict(model_save_path, stock, seq_len, x_test, y_test, x_tomorrow, date_time, x_close)

def _predict(model_save_path, stock, seq_len, x_test, y_test, x_tomorrow, date_time, x_close) :
    try :
        model = load_model(model_save_path, stock, seq_len)
        test_model(model, x_test, y_test, stock)
        predict_tomorrow(model, stock, x_tomorrow, date_time, x_close)
    except Exception as e :
        print(e)


if __name__ == "__main__":
    main()

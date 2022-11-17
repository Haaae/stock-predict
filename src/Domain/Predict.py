from datetime import datetime
import warnings

from Predict_Stock.Predictor import predict_tomorrow
from Predict_Stock.Loader import load_model
from src.Domain.Train_Model.Data_Loader import load_data
from src.Domain.Train_Model.Tester import test_model
from src.Domain.Train_Model.Stock_Loader import load_stocks

def main():
    warnings.filterwarnings("ignore")

    # Hyper Parameter
    date_time = datetime.now().strftime("%Y_%m_%d")   # 학습 시작 시간
    start = datetime(2015, 1, 1)    # 주식 기간 2015
    end = datetime.today()     # 주식 기간
    seq_len = 5
    test_rate = 0.2
    stocks_path_10per = "C:\\Users\\82106\\Desktop\\Stock\\Predict\\Stocks_trained_testRate_10per.txt"
    stocks_path = "C:\\Users\\82106\\Desktop\\Stock\\Predict\\Stocks_trained_testRate_20per.txt"

    # stocksPath_testRate = {stocks_path_10per:0.1, stocks_path_20per:0.2}

    # 2022-11-16 기준
    stocks = load_stocks(stocks_path)

    #print("test rate : " + test_rate)

    for stock in stocks:
        print('\n\n')
        print("*** " + stock + " Predict Start ***")
        stock_ticker = stocks[stock]
        model_save_path = "C:\\Users\\82106\\Desktop\\Stock\\Predict\\Model\\" + stock

        # Load dataset
        x_train, x_test, y_train, y_test, x_tomorrow, x_close = load_data(stock_ticker, start, end, seq_len, test_rate)

        _predict(stock, seq_len, x_test, y_test, x_tomorrow, date_time, x_close)

def _predict(stock, seq_len, x_test, y_test, x_tomorrow, date_time, x_close) :
    try :
        model = load_model(stock, seq_len)
        test_model(model, x_test, y_test, stock)
        predict_tomorrow(model, stock, x_tomorrow, date_time, x_close)
    except Exception as e :
        print(e)


if __name__ == "__main__":
    main()

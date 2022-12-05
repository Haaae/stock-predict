import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

global number_of_feature
global close_row_index
number_of_feature = 5
close_row_index = 3

def load_data(stock, start, end, seq_len, test_rate):

    # 주식 데이터 가져오기
    df = _get_data(stock, start, end)
    print("Dataset length : ", len(df))
    # 데이터 전처리
    x_train, x_test, y_train, y_test, x_tomorrow, x_close = _preprocessing(df, seq_len, test_rate)
    
    return x_train, x_test, y_train, y_test, x_tomorrow, x_close

# 주식 데이터 가져오기
def _get_data(stock, start, end):

    df = yf.download(stock, start=start, end=end)
    df.drop(['Adj Close'], axis=1, inplace=True)
    return df

# df 형식의 주식 데이터 전처리
def _preprocessing(df, seq_len, test_rate):
    
    x_train, x_test, y_train, y_test, x_tomorrow, x_close  = _data_split(df, seq_len, test_rate)
    x_train, x_test, x_tomorrow = _scaling(x_train, x_test, x_tomorrow)


    print("x_train length : ", x_train.shape[0])
    print("x_test length  : ", x_test.shape[0])

    x_train = x_train.reshape(x_train.shape[0], seq_len, number_of_feature)
    x_test = x_test.reshape(x_test.shape[0], seq_len, number_of_feature)
    x_tomorrow = x_tomorrow.reshape(x_tomorrow.shape[0], seq_len, number_of_feature)

    return x_train, x_test, y_train, y_test, x_tomorrow, x_close

# df 주식데이터를 x_traing, x_teat, y_train, y_test로 만들어 반환
def _data_split(df, seq_len, test_rate):

    data_X, data_Y, x_tomorrow, x_close = _split_stock(df, seq_len)

    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=test_rate, random_state=1, shuffle=False)
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_test.shape[2])

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

    x_tomorrow = x_tomorrow.reshape(x_tomorrow.shape[0], x_tomorrow.shape[1]*x_tomorrow.shape[2])
    
    return x_train, x_test, y_train, y_test, x_tomorrow, x_close

# df 형식의 주식 데이터로 train set과 teat set 만들어 반환
def _split_stock(df, seq_len):  # seq_len = time_step
    x, y, tomorrow = list(), list(), list()
    x_close = []
    for i in range(len(df)):
        x_end = i + seq_len
        y_end = x_end + 1
        
        if y_end > len(df):
            x_tomorrow = df.iloc[i:x_end, :]
            x_close = df.iloc[x_end-1:x_end, close_row_index]
            tomorrow.append(x_tomorrow)
            break
        
        x_tmp = df.iloc[i:x_end, :]
        y_tmp = df.iloc[x_end:y_end, close_row_index]
        x.append(x_tmp)
        y.append(y_tmp)
    
    return np.array(x), np.array(y), np.array(tomorrow), str(x_close.item())

# x_data 정규화 -1 ~ 1
def _scaling(x_train, x_test, x_tomorrow):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_tomorrow = scaler.transform(x_tomorrow)
    
    return x_train, x_test, x_tomorrow
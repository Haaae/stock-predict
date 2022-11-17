import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def test_model(model, x_test, y_test, stock):
    y_pred = model.predict(x_test, verbose=10)
    # compute and print the CC & NRMSE
    _print_CC_RMSE(y_pred, y_test)
    # show plot
    _show_plot(y_test, y_pred, stock)
    
def _show_plot(y_test, y_pred, stock):

    plt.plot(y_test, label='real')
    plt.plot(y_pred, label='forecast')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(stock)
    plt.legend()
    plt.show()
    
def _print_CC_RMSE(y_pred, y_test) :
    print("========================================================")
    print('CC   SCORE : ', _CC(y_pred, y_test)[0][1])  # 피어슨 상관계수
    
    print('NRMSE SCORE : ', _NRMSE(y_pred, y_test))
    print("========================================================")
    
def _CC(true, pred) :
    true = np.squeeze(np.asarray(true))
    pred = np.squeeze(np.asarray(pred))
    return np.corrcoef(true, pred)

def _NRMSE(true, pred) :
    scaler = MinMaxScaler()
    scaler.fit(true)
    scaled_true = scaler.transform(true)
    scaled_pred = scaler.transform(pred)
    return sklearn.metrics.mean_squared_error(scaled_true, scaled_pred)**0.5
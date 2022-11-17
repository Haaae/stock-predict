import numpy as np

def predict_tomorrow(model, stock,x_tomorrow, date_time, x_close):
    y_tomorrow = model.predict(x_tomorrow)
    y_tomorrow = np.array2string(y_tomorrow)[2:-2]
    
    # 상승 백분율 계산
    x = float(x_close)
    y = float(y_tomorrow)
    percent = (y - x) / x * 100
    
    file_path = 'C:\\Users\\82106\\Desktop\\Stock\\Predict\\Prediction_Result\\' + stock + '_Prediction_Close.txt'
    
    lines = '['+ date_time + ']\n' \
            + '현재가 : ' \
            + x_close + '\n' \
            + '다음 개장일 예측 종가 : ' \
            + y_tomorrow  + '\n' \
            + '상승 퍼센트 : ' \
            + str(percent) \
            + '%' \
            + '\n\n\n'
            
    with open(file_path, 'a', encoding='utf-8') as File:
        File.write(lines)
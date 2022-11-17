def load_stocks(stocks_path) :
    '''
    Kweichow Moutai:600519.SS
    ICBC:1398.HK // cc값은 높은데 nrmse값은 0.15 이 모델로 매매해도 괜찮을까?
    China Merchants Bank Co Ltd:CIHKY
    PetroChina Company:601857.SS
    삼성전자:005930.KS
    LG에너지솔루션:373220.KS
    SK하이닉:000660.KS
    삼성바이오로직스:207940.KS
    삼성SDI:006400.KS
    Apple:AAPL
    MicroSoft:MSFT
    Alpha C:GOOG
    Alpha A:GOOGL
    Amozon :AMZN
    '''
    with open(stocks_path, 'r', encoding="UTF-8") as f:
        stocks = {}

        for line in f:
            [stock, ticker] = line.strip().split(':')
            stocks[stock] = ticker

        return stocks
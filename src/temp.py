import FinanceDataReader as fdr

def main() :

       # 애플(AAPL), 2018-01-01 ~ 2018-03-30
       df = fdr.DataReader('AAPL', '2018-01-01', '2018-03-30')
       df.tail()

if __name__ == "__main__":
    main()
       


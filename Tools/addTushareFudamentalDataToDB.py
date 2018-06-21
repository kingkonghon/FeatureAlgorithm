import tushare as ts

def downloadTushareDataToDB():
    ts.get_stock_basics()
    pass


if __name__ == '__main__':
    downloadTushareDataToDB()
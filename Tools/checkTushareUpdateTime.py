import tushare
from datetime import datetime
import time

if __name__ == '__main__':
    while True:
        data = tushare.get_k_data('000001', ktype='D', autype='hfq', index=False, start='2018-04-26', end='2018-04-27')
        if data.shape[0] == 2:
            break
        else:
            time.sleep(5)

    print(datetime.now())
import quandl
from datetime import datetime
import time

quandlAuthenCode = 'n1Wbo94VP2FN9yRVN2iy'
# quandlProductCode = 'CHRIS/ICE_T1'
quandlProductCode = 'NASDAQOMX/COMP'

if __name__ == '__main__':
    quandl.ApiConfig.api_key = quandlAuthenCode

    # today = datetime.strftime(datetime.now(), '%Y-%m-%d')
    today = '2018-05-04'

    while True:
        try:
            data = quandl.get(quandlProductCode, start_date=today)
        except:
            print('error...', datetime.now())
            time.sleep(300)
            continue

        if not data.empty:
            break
        else:
            print('waiting...', datetime.now())
            time.sleep(60)

    print('succeed!', datetime.now())
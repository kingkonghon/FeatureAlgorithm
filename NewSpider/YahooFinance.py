import requests
from lxml import etree
import pandas as pd
from datetime import datetime
import sys
import os
from sqlalchemy import create_engine
import random
import time
from pymysql.err import ProgrammingError

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant

class MySpider:
    def __init__(self):
        self.target_url = {
            '.IXIC': "https://finance.yahoo.com/quote/%5EIXIC/history/",
            '.INX': "https://finance.yahoo.com/quote/%5EGSPC/history/",
            '.DJI': "https://finance.yahoo.com/quote/%5EDJI/history/"
        }

        self.headers_list = [
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36']


        self.quote_xpath_loc = {
            'date': "//td[@class='Py(10px) Ta(start) Pend(10px)']/span/text()",
            'open': "//td[@class='Py(10px) Pstart(10px)'][1]/span/text()",  # open price
            'high': "//td[@class='Py(10px) Pstart(10px)'][2]/span/text()",
            'low': "//td[@class='Py(10px) Pstart(10px)'][3]/span/text()",
            'close': "//td[@class='Py(10px) Pstart(10px)'][4]/span/text()",
            'volume': "//td[@class='Py(10px) Pstart(10px)'][6]/span/text()"
        }

        self.sql_table_name = 'YahooFinance'

    def run(self):
        for tmp_symbol, tmp_url in self.target_url.items():
            header = {'User-Agent': random.choice(self.headers_list)}
            page_res = requests.get(tmp_url, headers=header)

            spider_data = self.parse_page(tmp_symbol, page_res)

            self.write_sql(tmp_symbol, spider_data)

            time.sleep(5)  # sleep before crawling next url

    def parse_page(self, symbol, page_res):
        page_html = page_res.text

        url_parse = etree.HTML(page_html)

        spider_quote = {}

        for tmp_k, tmp_v in self.quote_xpath_loc.items():
            spider_quote[tmp_k] = url_parse.xpath(tmp_v) # extract data from html text

        spider_quote = pd.DataFrame(spider_quote)

        spider_quote.loc[:, 'date'] = spider_quote['date'].apply(lambda x: datetime.strptime(x, '%b %d, %Y'))
        spider_quote.loc[:, 'date'] = spider_quote['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d')) # change date format

        for tmp_col in ['open', 'high', 'low', 'close', 'volume']:
            spider_quote.loc[:, tmp_col] = spider_quote[tmp_col].apply(lambda x: x.replace(',', ''))
            spider_quote.loc[:, tmp_col] = spider_quote[tmp_col].astype('float')

        spider_quote = spider_quote.sort_values('date',ascending=True)

        spider_quote.loc[:, 'symbol'] = symbol
        spider_quote.loc[:, 'time_stamp'] = datetime.now()

        return spider_quote

    def write_sql(self, symbol, data):
        spider_engine = create_engine(
            'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

        sql_statement = "select max(date) from `%s` where `symbol` = '%s'" % (self.sql_table_name, symbol)

        sql_conn = spider_engine.connect()
        try:
            record_max_date = pd.read_sql(sql_statement, sql_conn)
            record_max_date = record_max_date.iloc[0, 0] # only one element

            if record_max_date is not None: # exists past data
                data = data.loc[data['date'] > record_max_date]
        except Exception as e:  # table not exist
            print(e)

        if not data.empty:
            data.to_sql(self.sql_table_name, sql_conn, index=False, if_exists='append') # add data to database
            print('%s: successful update new data!' % symbol)
        else:
            print('%s: not need to update' % symbol)

        sql_conn.close()

if __name__ == '__main__':
    ms = MySpider()
    ms.run()


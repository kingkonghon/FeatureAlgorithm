import requests
from lxml import etree
import pandas as pd
from datetime import datetime

#response = requests.get("https://hq.sinajs.cn/?_=0.2857763894050549&list=gb_$ixic,gb_ixic,sys_time,gb_$ixic_i,gb_ixic_i,sys_time_i")

url = "https://finance.yahoo.com/quote/%5EIXIC/history/"

xpath_loc ={
    'date': "//td[@class='Py(10px) Ta(start) Pend(10px)']/span/text()",
    'open': "//td[@class='Py(10px) Pstart(10px)'][1]/span/text()", # open price
    'high': "//td[@class='Py(10px) Pstart(10px)'][2]/span/text()",
    'low': "//td[@class='Py(10px) Pstart(10px)'][3]/span/text()",
    'close': "//td[@class='Py(10px) Pstart(10px)'][4]/span/text()",
    'volume': "//td[@class='Py(10px) Pstart(10px)'][6]/span/text()"
}

res = requests.get(url)
html = res.text

url_parse = etree.HTML(res.text)

spider_quote = {}

for tmp_k, tmp_v in xpath_loc.items():
    spider_quote[tmp_k] = url_parse.xpath(tmp_v)

spider_quote = pd.DataFrame(spider_quote)

spider_quote.loc[:, 'date'] = spider_quote['date'].apply(lambda x: datetime.strptime(x, '%b %d, %Y'))
spider_quote.loc[:, 'date'] = spider_quote['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))

for tmp_col in ['open', 'high', 'low', 'close', 'volume']:
    spider_quote.loc[:, tmp_col] = spider_quote[tmp_col].replace({',': ''})

print(spider_quote)
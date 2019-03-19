#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @Time    : 17-12-3 上午12:23
# @Author  : LiuXin
# @Site    :
# @File    : shibor数据.py
# @Software: PyCharm
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')      # 为了ｕｎｉｃｏｄｅ格式问题
import re
import time,random
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from time import sleep

import requests
# 导入库
# from scrapy import log
from lxml import etree
from retrying import retry
import pymysql
# import MySQLdb.cursors

DEBUG = False
if DEBUG:
    dbuser = 'quant'
    dbpass = '9974'
    dbname = 'spider_data'
    dbhost = '192.168.1.7'
    dbport = '3306'
else:
    dbuser = 'quant'
    dbpass = '9974'
    dbname = 'spider_data'
    dbhost = 'localhost'
    dbport = '3306'




class ShiBor(object):
    def __init__(self, sp_name=None):  # 去想要爬取的网站上寻找相关规律
        self.sp_name = sp_name
        self.conn = pymysql.connect(user=dbuser, passwd=dbpass, db=dbname, host=dbhost, charset="utf8",
                                    use_unicode=True)
        self.cursor = self.conn.cursor()
        # 清空表：
        # self.cursor.execute("truncate table shibor;")
        # self.conn.commit()
        self.logfile = open('./shibor数据.txt', 'ab+')

        self.temp_urls = 'http://www.shibor.org/shibor/web/html/shibor.html'
        # self.temp_urls = 'http://www.shibor.org/shibor/web/html/index.html'

        headers_list = [{'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'},{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; …) Gecko/20100101 Firefox/57.0'}]
        # 模拟浏览器设置
        self.headers = random.choice(headers_list)


    #@retry(stop_max_attempt_number=60)
    def _parse_url(self, url):  # 模拟浏览器访问网站　获取内容　并解析
        print('正在获取响应，爬取网页%s' % url)
        # response = requests.get(url=url, headers=self.headers, timeout=10,cookies=self.cookie)
        response = requests.get(url=url, headers=self.headers, timeout=30)

        datas = etree.HTML(response.content)
        html = datas.xpath("//tr[2]//td/table[2]//tr[position()<last()]")
        report_time = datas.xpath("//td[@class='infoTitleW']/text()")[0]
        print(html)
        return html,report_time

    def parse_url(self,url): #捕捉异常

        while True:
            try:
                html,report_time = self._parse_url(url)
                break
            except Exception as e:
                print(e)
                print('request url failed, retry after 60 seconds...')
                sleep(60)
                # html = None
        return html,report_time



    # def save_html(self, html, page_num):
    def save_html(self, html,url,report_time):
        # 保存之前先定义文件路径以及存储的文件名
        # 保存数据库
        info_from_url = url


        # print html
        # '报告日期',
        ReportDate = report_time[:16]


        # '期限O/N(shibor(%)利率)',
        TimeLimitO_N = html[0].xpath(".//td[3]/text()")[0]

        # 获取上涨下跌符号
        # '期限O/N(shibor涨跌)',
        TimeLimitO_N_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[0].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimitO_N_BP = html[0].xpath(".//td[5]/text()")[0].strip()
        if TimeLimitO_N_BPfuhao == 'downicon':
            TimeLimitO_N_BP = '-'+TimeLimitO_N_BP


        # '期限1W(shibor(%)利率)',
        TimeLimit1W = html[1].xpath(".//td[3]/text()")[0]

        # 获取上涨下跌符号
        # '期限1W(shibor涨跌)',
        TimeLimit1W_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[1].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimit1W_BP = html[1].xpath(".//td[5]/text()")[0].strip()
        if TimeLimitO_N_BPfuhao == 'downicon':
            TimeLimit1W_BP = '-'+TimeLimit1W_BP

        # '期限2W(shibor(%)利率)',
        TimeLimit2W = html[2].xpath(".//td[3]/text()")[0]


        # 获取上涨下跌符号
        # '期限2W(shibor涨跌)',
        TimeLimit2W_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[2].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimit2W_BP = html[2].xpath(".//td[5]/text()")[0].strip()
        if TimeLimit2W_BPfuhao == 'downicon':
            TimeLimit2W_BP = '-'+TimeLimit2W_BP

        # '期限1M(shibor(%)利率)',
        TimeLimit1M = html[3].xpath(".//td[3]/text()")[0]

        # 获取上涨下跌符号
        # '期限1M(shibor涨跌)',
        TimeLimit1M_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[3].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimit1M_BP = html[3].xpath(".//td[5]/text()")[0].strip()
        if TimeLimit1M_BPfuhao == 'downicon':
            TimeLimit1M_BP = '-'+TimeLimit1M_BP

        # '期限3M(shibor(%)利率)',
        TimeLimit3M = html[4].xpath(".//td[3]/text()")[0]

        # 获取上涨下跌符号
        # '期限3M(shibor涨跌)',
        TimeLimit3M_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[4].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimit3M_BP = html[4].xpath(".//td[5]/text()")[0].strip()
        if TimeLimit3M_BPfuhao == 'downicon':
            TimeLimit3M_BP = '-'+TimeLimit3M_BP

        # '期限6M(shibor(%)利率)',
        TimeLimit6M = html[5].xpath(".//td[3]/text()")[0]

        # 获取上涨下跌符号
        # '期限6M(shibor涨跌)',
        TimeLimit6M_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[5].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimit6M_BP = html[5].xpath(".//td[5]/text()")[0].strip()
        if TimeLimit6M_BPfuhao == 'downicon':
            TimeLimit6M_BP = '-'+TimeLimit3M_BP


        # '期限9M(shibor(%)利率)',
        TimeLimit9M = html[6].xpath(".//td[3]/text()")[0]

        # 获取上涨下跌符号
        # '期限9M(shibor涨跌)',
        TimeLimit9M_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[6].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimit9M_BP = html[6].xpath(".//td[5]/text()")[0].strip()
        if TimeLimit9M_BPfuhao == 'downicon':
            TimeLimit9M_BP = '-'+TimeLimit9M_BP

        # '期限1Y(shibor(%)利率)',
        TimeLimit1Y = html[7].xpath(".//td[3]/text()")[0]

        # 获取上涨下跌符号
        # '期限1Y(shibor涨跌)',
        TimeLimit1Y_BPfuhao = re.findall(r'(?<=newimages\/).*(?=\.gif)',html[7].xpath(".//td[4]/img/@src")[0])[0]
        TimeLimit1Y_BP = html[7].xpath(".//td[5]/text()")[0].strip()
        if TimeLimit1Y_BPfuhao == 'downicon':
            TimeLimit1Y_BP = '-'+TimeLimit1Y_BP


        time_stamp = time.strftime('%Y年%m月%d日 %H:%M:%S',time.localtime(time.time()))

        self.cursor.execute("select * from shibor where ReportDate ='%s'" % (ReportDate))
        ret = self.cursor.fetchone()
        if ret:
            pass
        else:

            try:
                sql = (
                "insert into shibor(ReportDate,TimeLimitO_N,TimeLimitO_N_BP,TimeLimit1W,TimeLimit1W_BP,"
                "TimeLimit2W,TimeLimit2W_BP,TimeLimit1M,TimeLimit1M_BP,TimeLimit3M,TimeLimit3M_BP,"
                "TimeLimit6M,TimeLimit6M_BP,TimeLimit9M,TimeLimit9M_BP,TimeLimit1Y,TimeLimit1Y_BP,"
                "time_stamp,info_from_url) ""values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
                param = (ReportDate,TimeLimitO_N,TimeLimitO_N_BP,TimeLimit1W,TimeLimit1W_BP,TimeLimit2W,TimeLimit2W_BP,TimeLimit1M,TimeLimit1M_BP,TimeLimit3M,TimeLimit3M_BP,TimeLimit6M,TimeLimit6M_BP,TimeLimit9M,TimeLimit9M_BP,TimeLimit1Y,TimeLimit1Y_BP,time_stamp,info_from_url)



                print(param)
                self.cursor.execute(sql, param)
                self.conn.commit()
                print("数据存库成功")

            except Exception as e:
                print(e)
                self.logfile.write("Error : %s %s" % (e))




        # 保存本地文件
        # file_path =  './财务数据' + str(page_num) + '.html'
        # with open(file_path, 'ab+', ) as f:
        #     f.write(html)

    def run(self):



        # 1　寻找ｕｒｌ的规律
        # 2. 匹配ｕｒｌ后的ｈｔｍｌ，发送请求获取相应　采用ｇｅｔ方式
        # 3. 获取爬取道的字符串

        html,report_time = self.parse_url(self.temp_urls)

        # 4. 保存爬取到的内容到指定文件目录和文明名
        # page_num = self.url_list.index(url)  # 获取页码数
        self.save_html(html,self.temp_urls,report_time)

        print('save successful')
        # 关闭log记录　数据库链接
        self.logfile.close()
        self.cursor.close()
        self.conn.close()
        print ('log保存成功')
        pass

def supplement_by_file():
    file_path = r'F:\Shibor_2019_1.xls'

    raw_data = pd.read_excel(file_path, encoding='gbk')
    source_fields = ['日期', 'O/N', '1W', '2W', '1M', '3M', '6M', '9M', '1Y']
    targget_fields = ['ReportDate', 'TimeLimitO_N', 'TimeLimit1W', 'TimeLimit2W', 'TimeLimit1M', 'TimeLimit3M', 'TimeLimit6M', 'TimeLimit9M', 'TimeLimit1Y']
    chg_name_dict = dict(zip(source_fields, targget_fields))
    raw_data = raw_data.rename(chg_name_dict, axis=1)  # change name

    raw_data.loc[:, 'ReportDate'] = raw_data['ReportDate'].apply(
        lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M'))  # change date format

    # get the existing record dates
    tmp_config = {'host': '123.123.136.57',
                   'user': 'quant',
                   'password': '9974',
                   'db': 'spider_data',
                   # 'port': 3306,
                   'charset': 'utf8'}
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**tmp_config))
    sql_conn = quant_engine.connect()
    sql_statement = 'select `ReportDate` from shibor'
    existing_record_date = pd.read_sql(sql_statement, sql_conn)
    existing_record_date = existing_record_date['ReportDate'].tolist()

    new_records = raw_data.loc[~raw_data['ReportDate'].isin(existing_record_date)]

    for tmp_idx in new_records.index:
        tmp_record = new_records.loc[tmp_idx].tolist()

        sql_statement = "insert into shibor(ReportDate,TimeLimitO_N,TimeLimit1W, TimeLimit2W,TimeLimit1M,TimeLimit3M, TimeLimit6M,TimeLimit9M,TimeLimit1Y) values('%s','%s','%s','%s','%s','%s','%s','%s','%s')" % tuple(tmp_record)
        sql_conn.execute(sql_statement)

    sql_conn.close()



def main():
    # spider = ShiBor()
    # spider.run()
    supplement_by_file()


if __name__ == "__main__":
    main()

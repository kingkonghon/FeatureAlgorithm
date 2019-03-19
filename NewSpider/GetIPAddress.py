import re
import requests
import traceback
import sys, getopt
from urllib.parse import quote
from lxml import etree
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime


class crawler:
    url = u''
    urls = []
    o_urls = []
    html = ''
    total_pages = 5
    current_page = 0
    next_page_url = ''
    timeout = 60                    #默认超时时间为60秒
    headersParameters = {    #发送HTTP请求时的HEAD信息，用于伪装为浏览器
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Mozilla/6.1 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
    }

    adminMailConfig = [
        {
            'smtp_server': 'smtp.163.com',
            'smtp_port': 465,
            'user': '13602819622@163.com',
            'password': 'll900515',

            'sender': '13602819622@163.com',
            'receiver': '13602819622@163.com',
        }
    ]

    def __init__(self, keywork, xpath_loc):
        self.url = 'https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=0&rsv_idx=1&tn=baidu&wd=%20' + quote(keywork)
        self.xpath_loc = xpath_loc

    def sendEmail(self, mail_content, mail_config):
        if mail_content['content'] != '':
            try:
                # smtpObj = smtplib.SMTP()  # send email
                # smtpObj.connect(mail_config['smtp_server'], mail_config['smtp_port'])
                smtpObj = smtplib.SMTP_SSL()
                smtpObj.connect(mail_config['smtp_server'])
                smtpObj.login(mail_config['user'], mail_config['password'])

                # construct message
                message = MIMEText(mail_content['content'], 'plain', 'utf-8')
                message['From'] = Header(mail_content['from'], 'utf-8')
                message['To'] = Header(mail_content['to'], 'utf-8')
                message['Subject'] = Header(mail_content['subject'], 'utf-8')

                # send E-mail
                smtpObj.sendmail(mail_config['sender'], mail_config['receiver'], message.as_string())
            except smtplib.SMTPException:
                print('send email to %s error' % mail_config['receiver'])


    def run(self):
        html = requests.get(self.url, timeout=self.timeout, headers=self.headersParameters)
        if html.status_code == 200:
            html = html.text
        else:
            print('get html error')
            raise Exception
        parsed_html = etree.HTML(html)

        ip = parsed_html.xpath(self.xpath_loc)
        ip = ip[0]
        tmp_str = '本机IP:\xa0'
        ip = ip[len(tmp_str):]

        # write email
        mail_content = {}
        mail_content['content'] = ip
        mail_content['from'] = 'IP Reporter'
        mail_content['to'] = 'Subscriber'
        mail_content['subject'] = 'IP Update (%s)' % (datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M'))

        for tmp_config in self.adminMailConfig:
            self.sendEmail(mail_content, tmp_config)



if __name__ == '__main__':
    xp_loc = r"//td/span[@class='c-gap-right']/text()"
    cl = crawler('ip', xp_loc)
    cl.run()

import os
import sys

from tableConfigMorning import tableConfig, tableDirection
from generalCheckDataUpdateFunctions import sendErrorByEmail, checkTodayAllTableRecordNum
from specialCheckDataUpdate import specialRecordCheckMorning

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

# ==============  Email config ==================
mailConfig = {
    'smtp_server': 'smtp.mxhichina.com',
    'smtp_port': 25,
    'user': 'jianghan@nuoyuan.com.cn',
    'password': 'jh@880528',

    'sender': 'jianghan@nuoyuan.com.cn',
    'receiver': 'jianghan@nuoyuan.com.cn',
}

if __name__ == '__main__':
    error_tables = checkTodayAllTableRecordNum(tableConfig, tableDirection, ConfigQuant)
    error_tables.extend( specialRecordCheckMorning() )
    sendErrorByEmail(error_tables, mailConfig)
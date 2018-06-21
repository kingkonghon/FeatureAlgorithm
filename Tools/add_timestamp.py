from sqlalchemy import create_engine
import pymysql

ConfigQuant = {'host':'10.46.228.175',
                     'user':'root',
                     'password':'xunzhaoshengbei',
                     'db':'quant',
                     'charset':'utf8'}

add_colume_statement = "ALTER TABLE %s ADD COLUMN time_stamp TIMESTAMP NOT NULL AFTER `%s`"

update_value_statement = "UPDATE %s SET time_stamp = TIMESTAMP('%s')"


if __name__ == '__main__':
    table_name = ['guijinshu', 'shibor', 'yucezhe_market_index', 'yucezhe_market_overviewdata']
    last_column_name = ['AMOUNT', 'Transaction_Amount', '1Y', 'type', 'amount_ratio']
    time_stamp_value = ['2017-10-23']

    db = pymysql.connect(ConfigQuant['host'], ConfigQuant['user'], ConfigQuant['password'], ConfigQuant['db'])
    cursor = db.cursor()

    for i, table in enumerate(table_name):
        tmp_statement = 'DESCRIBE %s' % table
        cursor.execute(tmp_statement) # get table column names
        tmp_data = cursor.fetchall()
        table_col_names = list(map(lambda x:x[0], tmp_data))
        if 'time_stamp' not in table_col_names: # have not added time_stamp
            tmp_statement = add_colume_statement % (table, last_column_name[i])
            cursor.execute(tmp_statement)

        tmp_statement = update_value_statement % (table, time_stamp_value[i])
        cursor.execute(tmp_statement)


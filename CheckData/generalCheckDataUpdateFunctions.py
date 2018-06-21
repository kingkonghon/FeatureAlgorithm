import smtplib
from email.mime.text import MIMEText
from email.header import Header
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime


calendarTableName = 'TRADE_CALENDAR'

def sendErrorByEmail(error_tables, mail_config):
    if error_tables != []:
        try:
            smtpObj = smtplib.SMTP()  # send email
            smtpObj.connect(mail_config['smtp_server'], mail_config['smtp_port'])
            smtpObj.login(mail_config['user'], mail_config['password'])

            # construct message
            str_error_tables = '\n'.join(error_tables)
            message = MIMEText('update failed table list:\n%s' % str_error_tables, 'plain', 'utf-8')
            message['From'] = Header('Stock data update checking algorithm', 'utf-8')
            message['To'] = Header('monitor', 'utf-8')
            message['Subject'] = Header('Stock data update error report', 'utf-8')

            # send E-mail
            smtpObj.sendmail(mail_config['sender'], mail_config['receiver'], message.as_string())
        except smtplib.SMTPException:
            print('send email to %s error' % mail_config['receiver'])

def getTableRecordNum(table_config, today):
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**table_config['db_conn_config']))

    try:
        str_today = datetime.strftime(today, table_config['date_format'])
    except UnicodeEncodeError:  #  chinese date format
        import locale
        locale.setlocale(locale.LC_CTYPE, 'chinese')
        str_today = datetime.strftime(today, table_config['date_format'])
    if table_config['distinct_fields'] != []:
        tmp_distinct_fields = ','.join(table_config['distinct_fields'])
        # fetch distinct data and calculate record num later
        sql_statement = "select distinct %s from %s where (%s = '%s')" % (tmp_distinct_fields, table_config['table_name'],
                                                                        table_config['date_field'], str_today)
    else:
        sql_statement = "select count(1) as num from %s where (%s = '%s')" % (table_config['table_name'],
                                                                  table_config['date_field'], str_today)
    if table_config['condition'] != '':
        sql_statement += ' and (%s)' % table_config['condition']

    record_num = pd.read_sql(sql_statement, sql_engine)  # read record num
    if table_config['distinct_fields'] != []:
        record_num = record_num.shape[0]  # record num of the distinct data
    else:
        record_num = record_num.iloc[0,0]

    return record_num


def checkTodayAllTableRecordNum(table_config, table_direction, sql_config):
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    # fetch calendar
    sql_statement = "select date from %s" % calendarTableName
    calendar = pd.read_sql(sql_statement, sql_engine)
    calendar = calendar['date'].values

    today = datetime.now()
    str_today = datetime.strftime(today, '%Y-%m-%d')

    error_tables = []
    if np.sum(calendar == str_today) == 1:  #  today is trade day
        # get db's record num
        table_record_num = {}
        for name, config in table_config.items():
            # offset the record date (spider crawl the next day)
            if config['offset_day_num'] != 0:
                real_today = calendar[np.where(calendar == str_today)[0][0] - config['offset_day_num']]
                real_today = datetime.strptime(real_today, '%Y-%m-%d')
            else:
                real_today = today
            table_record_num[name] = getTableRecordNum(config, real_today)

        # get topologic order of tables
        all_topo_paths = getAllTopoPaths(list(table_config.keys()), table_direction)

        for topo_path in all_topo_paths:  #  loop over all topo path
            if len(topo_path) <=1:
                print('topo path length is too short!')
                raise ValueError
            path_record_num = list(map(lambda x: table_record_num[x], topo_path))  # mapping table to table record num

            if path_record_num[0] == 0:  #  the starting point of a path has no today's data (spider failed)
                error_tables.append(topo_path[0])
            else:
                tmp_num = np.array(path_record_num[1:])
                tmp_tables = np.array(topo_path[1:])
                # find tables that have different record num
                tmp_error_tables = tmp_tables[tmp_num != path_record_num[0]]
                error_tables.extend(tmp_error_tables)

        # drop duplicates
        error_tables = list(set(error_tables))

    if error_tables != []:
        error_tables = list(map(lambda x: table_config[x]['table_name'], error_tables))

    return error_tables

def getAllTopoPaths(points, edges):
    all_topo_paths = []

    if len(points) != 0:
        # find start points
        starting_points = points.copy()
        for child_points in edges.values():
            tmp_points = np.array(starting_points)
            tmp_points = tmp_points[np.in1d(tmp_points, child_points)]
            starting_points = list(set(starting_points) - set(list(tmp_points)))  # remove child points from the set

        # if cannot find a start point, means there is a circle
        if starting_points == []:
            print('there is a circle in DAG')
            raise ValueError

        # find all paths for all starting point
        for tmp_start_point in starting_points:  # loop by start points
            tmp_paths = extendPath([tmp_start_point], edges)
            all_topo_paths.extend(tmp_paths)

    return all_topo_paths

def extendPath(path, edges, paths=None):
    if paths is None:
        paths = []
    last_point_in_path = path[-1]
    if last_point_in_path in edges.keys():  # the last point in path has child points
        for next_point in edges[last_point_in_path]: #  recursively add child point to the path
            new_path = path + [next_point]
            paths = extendPath(new_path, edges, paths) #  update buffer (recursively)
    else:  # reach the end point
        paths.append(path)  # add this complete path into the buffer
    return paths


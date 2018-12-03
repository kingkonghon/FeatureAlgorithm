import pandas as pd

def writeDB(sql_conn, table_name, data):
    # check if there are already records in the table
    new_dates = data['date'].unique()

    new_start_date = new_dates.min()
    new_end_date = new_dates.max()

    sql_statement = "select count(1) as num from `%s` where `date` between '%s' and '%s'" % (table_name, new_start_date, new_end_date)
    old_record_num = pd.read_sql(sql_statement, sql_conn)
    old_record_num = old_record_num.iloc[0, 0]

    if old_record_num > 0:
        print('delete old data from %s to %s' % (new_end_date, new_end_date))
        sql_statement = "delete from `%s` where `date` between '%s' and '%s'" % (table_name, new_start_date, new_end_date)
        sql_conn.execute(sql_statement)

    # write new data to db
    data.to_sql(table_name, sql_conn, index=False, if_exists='append')


def renameDF(df, old_col_name, new_col_name):
    rename_dict = {}
    for field_pair in zip(old_col_name, new_col_name):
        rename_dict[field_pair[0]] = field_pair[1]
    new_df = df.rename(columns=rename_dict)
    return new_df


def chgDFDataType(df, col_name, new_data_type):
    for col in col_name:
        df.loc[:, col] = df[col].astype(new_data_type)

    return df
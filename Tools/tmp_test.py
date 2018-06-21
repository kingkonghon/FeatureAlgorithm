import pandas as pd
from datetime import datetime

file_name = 'a.log'

time = []
sql_states = []

with open(file_name, 'r') as file:
    for line in file.readlines():
        line = line.strip()
        time.append(line[:19])
        sql_states.append(line[21:-1])

time = list(map(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), time))
df = pd.DataFrame({'time':time, 'sql_states':sql_states})
df.loc[:,'count'] = 1

df = df.pivot_table(values='count', index='time', columns='sql_states')
df = df.fillna(0)

df = df.resample('T').sum()

max_sql_state = df.idxmax(axis=1)
max_sql_state.name = 'max_sql_state'

print(max_sql_state)


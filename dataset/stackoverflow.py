import pandas as pd
# import torch
import datetime
import time
import numpy as np

# https://snap.stanford.edu/data/sx-stackoverflow.html

datapath = './data/dataset/stackoverflow/'

filepath = datapath + 'sx-stackoverflow.txt'

cite_df = pd.read_csv(filepath, delimiter=' ', header=None)

print(cite_df.head())
print(cite_df.describe())


# 1217567877 Fri Aug 01 2008 05:17:57 GMT+0000
# 1457273428 Sun Mar 06 2016 14:10:28 GMT+0000

start_unixtime = cite_df[2][0]
end_unixtime = cite_df[2][len(cite_df) - 1]

# Convert the Unix timestamp to a datetime object
start_time = datetime.datetime.fromtimestamp(start_unixtime).replace(hour=0, minute=0, second=0, microsecond=0)
end_time = datetime.datetime.fromtimestamp(end_unixtime).replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)

start_unixtime = time.mktime(start_time.timetuple())
deta_unixtime = time.mktime((start_time + datetime.timedelta(days=1)).timetuple()) - start_unixtime
end_unixtime = time.mktime(end_time.timetuple())

delete_epoch = [0] * 10

for t in cite_df[2]:
    ix = int((t - start_unixtime) / deta_unixtime)
    if ix < 10:
        delete_epoch[ix] += 1
    else:
        break

insert_epoch = [0] * 10

for t in cite_df[2][::-1]:
    ix = 9 - int((end_unixtime - t) / deta_unixtime)
    if ix >= 0:
        insert_epoch[ix] += 1
    else:
        break

df_describe = pd.DataFrame(delete_epoch)
print(df_describe.describe())

df_describe = pd.DataFrame(insert_epoch)
print(df_describe.describe())

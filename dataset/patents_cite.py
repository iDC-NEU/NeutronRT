import pandas as pd
import torch

# https://www.nber.org/research/data/us-patents-1975-1999

newdatapath = './data/patents_cite/'

# cite_filepath = './data/cite75_99.txt'
# patents_filepath = './data/apat63_99.txt'

# cite_df = pd.read_csv(cite_filepath, delimiter=',')

# print(cite_df.head())

# patents_df = pd.read_csv(patents_filepath, delimiter=',')

# print(patents_df.head())

# max_cite = max(cite_df['CITING'].max(), cite_df['CITED'].max())
# min_cite = min(cite_df['CITING'].min(), cite_df['CITED'].min())

# print(max_cite, min_cite)

# patents_df = patents_df[(patents_df['PATENT'] >= min_cite) & (patents_df['PATENT'] <= max_cite)]

# print(patents_df.head())

# patents_df = patents_df.drop(columns=['GDATE', 'APPYEAR', 'COUNTRY', 'POSTATE'], axis=1)
# patents_df = patents_df.fillna(0)

# print(patents_df.head())
# print(patents_df.tail())

# min_pat = patents_df['PATENT'].min()
# max_pat = patents_df['PATENT'].max()

# cite_df = cite_df[(cite_df['CITING'] >= min_pat) & (cite_df['CITING'] <= max_pat) & (cite_df['CITED'] >= min_pat) & (cite_df['CITED'] <= max_pat)]

# print(cite_df.head())

# cite_df['CITING'] = cite_df['CITING'] - min_pat
# cite_df['CITED'] = cite_df['CITED'] - min_pat
# patents_df['PATENT'] = patents_df['PATENT'] - min_pat

# print(cite_df.head())
# print(patents_df.head())

# nclass_type = {}
# index_type = 0
# for i in patents_df['NCLASS']:
#     if i not in nclass_type:
#         nclass_type[i] = index_type
#         index_type += 1

# print(len(nclass_type))

# patents_df['NCLASS'] = patents_df['NCLASS'].apply(lambda x: nclass_type[x])

# print(patents_df.head())

# cite_df.to_csv(newdatapath + 'cite75_99.csv', index=False)
# patents_df.to_csv(newdatapath + 'pat63_99.csv', index=False)

# hash_id = {}
# for index, id in patents_df['PATENT'].items():
#     hash_id[id] = index

# drop_index = []
# for index, row in cite_df.iterrows():
#     if row['CITING'] not in hash_id or row['CITED'] not in hash_id:
#         drop_index.append(index)

# cite_df = cite_df.drop(drop_index, axis=0)

# print(cite_df.tail())

# patents_df['PATENT'] = patents_df['PATENT'].apply(lambda x: hash_id[x])
# cite_df['CITING'] = cite_df['CITING'].apply(lambda x: hash_id[x])
# cite_df['CITED'] = cite_df['CITED'].apply(lambda x: hash_id[x])

# nclass_type = {}
# index_type = 0
# for i in patents_df['CAT']:
#     if i not in nclass_type:
#         nclass_type[i] = index_type
#         index_type += 1

# print(len(nclass_type))

# patents_df['CAT'] = patents_df['CAT'].apply(lambda x: nclass_type[x])

# print(patents_df.tail())

# cite_df.to_csv(newdatapath + 'cite75_99.csv', index=False)
# patents_df.to_csv(newdatapath + 'pat63_99.csv', index=False)

# year_index = patents_df.groupby('GYEAR')['PATENT'].max()

cite_filepath = newdatapath + 'cite75_99.csv'
patents_filepath = newdatapath + 'pat63_99.csv'

cite_df = pd.read_csv(cite_filepath, delimiter=',')

print(cite_df.head())

patents_df = pd.read_csv(patents_filepath, delimiter=',')

print(patents_df.tail())

# year, node_id, edge index
years = [[], [], []]

for year in range(1975, patents_df['GYEAR'].max() + 1):
    years[0].append(year)
    id = patents_df[patents_df['GYEAR'] == year]['PATENT'].max()
    years[1].append(id)
    index = cite_df[cite_df['CITING'] == id].index
    while len(index) <= 0:
        id -= 1
        index = cite_df[cite_df['CITING'] == id].index
    years[2].append(index[-1])

data_x = patents_df[patents_df.columns.difference(['PATENT'])]
# data_x = patents_df[patents_df.columns.difference(['PATENT', 'NCLASS'])]
data_x = torch.from_numpy(data_x.values).float()

data_y = torch.from_numpy(patents_df['NCLASS'].values).long() 
edge_index = torch.from_numpy(cite_df.values).t().long()

year_index = torch.tensor(years, dtype=torch.long)

output = {'x': data_x, 'y': data_y, 'edge_index': edge_index, 'year_index': year_index, 'num_classes': data_y.max() + 1, 'num_features': len(data_x[0])}

torch.save(output, newdatapath + 'patents_cite_all.pt')


data_x = patents_df[patents_df.columns.difference(['PATENT', 'CAT'])]
data_x = torch.from_numpy(data_x.values).float()

data_y = torch.from_numpy(patents_df['CAT'].values).long() 

output = {'x': data_x, 'y': data_y, 'edge_index': edge_index, 'year_index': year_index, 'num_classes': data_y.max() + 1, 'num_features': len(data_x[0])}
torch.save(output, newdatapath + 'patents_cite_cat.pt')

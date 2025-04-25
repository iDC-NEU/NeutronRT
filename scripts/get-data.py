import sys
import re

filename = 'logs/update-all-6-3.txt'

if len(sys.argv) > 1:
    filename = sys.argv[1]

outdict = None
out_list = ['update_emb_times', 'sample_times', 'transfer_times',
            'replace_times', 'compute_times', 'inference_times',
            'update_graph_times']
baseline = ['basic', 'cache', 'more cache', 'partly']
# baseline = ['cache', 'more cache', 'partly']

def output(outdict, out_list, baseline):
    if outdict is not None:
        print(' ', end=',')
        for b in baseline:
            print(b, end=',')
        print()
        for key in out_list:
            print(key, end=',')
            if key in outdict:
                for b in baseline:
                    if b in outdict[key]:
                        print(outdict[key][b], end=',')
                    else:
                        print('na', end=',')
            print()

with open(filename) as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        if line.startswith('Dataset'):
            output(outdict, out_list, baseline)
            outdict = {}
            print(line)
        elif line.startswith('+ '):
            base_name = line.replace('+ ', '')
        elif re.match(r'^[a-zA-Z]+_', line) is not None:
            items = line.split(': ')
            if items[0] not in outdict:
                outdict[items[0]] = {}
            outdict[items[0]][base_name] = items[1].split(', ')[-1]
    output(outdict, out_list, baseline)


import argparse

from pathlib import Path as P

keywords = [
    'Computation', 'Affect', 'Sample'
]

methods = [
    'SMP5', 'SMP10', 'FULL', 'CACHE', 'INC'
]

def log_handler(path: P):
    e, b = None, None
    with open(path, 'r') as fin:
        for line in fin.readlines():
            if line.find('End>') != -1:
                words = line.split('>')
                e = max(e, float(words[1])) if e is not None else float(words[1])
            elif line.find('Begin>') != -1:
                words = line.split('>')
                b = min(b, float(words[1])) if b is not None else float(words[1])
    print(f'{e - b:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'files', type=str, nargs='+',
        help='paths of the file to be parsed',
    )
    args = parser.parse_args()
    paths = [P(p) for p in args.files]

    print(paths)

    for path in paths:
        log_handler(path)


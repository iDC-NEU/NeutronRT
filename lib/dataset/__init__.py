from typing import Dict, Callable, Optional
from dgl.data import AsNodePredDataset, RedditDataset, DGLDataset
from ogb.nodeproppred import DglNodePropPredDataset

from .friendster import FriendsterDataset
from .twitter import TwitterDataset

dataset_loaders: Dict[str, Callable] = {
    'ogbn': lambda n, r: AsNodePredDataset(DglNodePropPredDataset(n, r), verbose=True),
    'reddit': lambda n, r: RedditDataset(raw_dir=r, verbose=True),
    'twitter': lambda n, r: TwitterDataset(raw_dir=r, verbose=True),
    'friendster': lambda n, r: FriendsterDataset(raw_dir=r, verbose=True),
}

def load_dataset(ds_name: str, ds_root: str) -> DGLDataset:
    loader: Optional[Callable] = None
    if ds_name.startswith('ogbn-'):
        loader =  dataset_loaders['ogbn']
    elif ds_name in dataset_loaders:
        loader = dataset_loaders[ds_name]
    else:
        raise ValueError(f'{ds_name} not supported')
    
    return loader(ds_name, ds_root)
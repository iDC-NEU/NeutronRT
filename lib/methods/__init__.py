
from typing import Dict

from .basic import Basic
from .cache import Cache
from .inc import Incremental
from .smp import Sample


method_map: Dict[str, Basic] = {
    'smp': Sample,
    'cache': Cache,
    'inc': Incremental,
}

Method = Basic

def load_method(method: str):
    return method_map[method]
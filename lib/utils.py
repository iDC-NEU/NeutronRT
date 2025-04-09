import torch, resource


def tensor_text(t: torch.Tensor) -> str:
    if isinstance(t, torch.Tensor):
        return f'{t.dtype}@{t.device} {t.shape}'
    else:
        return f'{t}'
    

def free_ram() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def last_bit_set(x: int) -> int:
    assert isinstance(x, int), 'bit op on non-integer'
    return 0 if x == 0 else x.bit_length()


def floor_log2(x: int) -> int:
    assert x > 0, 'log with neg num'
    return last_bit_set(x) - 1


def ceil_log2(x: int) -> int:
    assert x > 0, 'log with neg num'
    return last_bit_set(x - 1)


def floor_div(x: int, y: int) -> int:
    return x // y


def ceil_div(x: int, y: int) -> int:
    return 0 if x == 0 else 1 + (x - 1) // y


def hyper_floor(x: int) -> int:
    return 1 << floor_log2(x)


def hyper_ceil(x: int) -> int:
    return 1 << ceil_log2(x)



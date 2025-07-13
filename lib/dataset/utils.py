import torch as th

NUM_FEATURES = 128
NUM_CLASSES = 50

def dummy_feat(num_nodes: int, num_features: int):
    return th.rand(
        (num_nodes, num_features),
        dtype=th.float32,
    )


def dummy_label(num_nodes: int, num_classes: int):
    return th.randint(num_classes, (num_nodes,))
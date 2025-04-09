from .model import Basic
from .conv import SAGE, GIN, GCN, GAT

def model_factory(
        model_name: str,
        num_features: int,
        num_hiddens: int,
        num_classes: int,
        num_model_layers: int,
        **kwargs
):
    if model_name == 'sage':
        model = SAGE(num_features, num_hiddens, num_classes, num_model_layers, **kwargs)
    elif model_name == 'gin':
        model = GIN(num_features, num_hiddens, num_classes, num_model_layers, **kwargs)
    elif model_name == 'gcn':
        model = GCN(num_features, num_hiddens, num_classes, num_model_layers, **kwargs)
    elif model_name == 'gat':
        model = GAT(num_features, num_hiddens, num_classes, num_model_layers, **kwargs)
    else:
        raise NotImplementedError
    return model


class Task():
    def __init__(
            self,
            model: str,
            num_features: int,
            num_hiddens: int,
            num_classes: int,
            num_model_layers: int,
            **kwargs,
    ):
        self.model: Basic = model_factory(
            model_name=model,
            num_features=num_features,
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            num_model_layers=num_model_layers,
            **kwargs,
        )
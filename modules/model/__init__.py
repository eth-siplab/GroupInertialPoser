from .gcn_encoder import Graph_JP_estimator
from .uip import UIP
from .pip import PIP
from .SSM import SSM
# from .MAM import Mamba

MODELS = {"GNN_JP":Graph_JP_estimator ,"UIP":UIP, "PIP":PIP, "SSM": SSM}

def get_model(args, parser):
    model_cls = MODELS[args.network]
    model_cls.add_args(parser)
    return model_cls
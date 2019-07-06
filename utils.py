import torch
import torch.nn as nn
import blocks
import torch.nn.functional as F
import numpy as np
from random import randint
from graphs import GraphsTuple


def data_dicts_to_graphs_tuple(graph_dicts:dict):
    for k,v in graph_dicts.items():
        graph_dicts[k]=torch.tensor(v)
    return GraphsTuple(**graph_dicts)
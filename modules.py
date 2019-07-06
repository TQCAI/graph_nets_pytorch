import torch
import torch.nn as nn
import blocks
import torch.nn.functional as F
import numpy as np
from random import randint
from graphs import GraphsTuple

class GraphNetwork(nn.Module):
    def __init__(self,graph):
        super(GraphNetwork,self).__init__()
        self._edge_block = blocks.EdgeBlock(graph)
        self._node_block = blocks.NodeBlock(graph)
        self._global_block = blocks.GlobalBlock(graph)
    def forward(self, graph):
        return self._node_block(self._edge_block(graph))
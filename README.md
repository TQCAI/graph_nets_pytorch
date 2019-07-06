# Graph Nets implement by pytorch



[Graph Nets](https://github.com/deepmind/graph_nets) is DeepMind's library for
building graph networks in Tensorflow and Sonnet.You can see it in https://github.com/deepmind/graph_nets

I have implemented `Graph Nets` by `Pytorch` framework. You can see my work in https://github.com/TQCAI/graph_nets_pytorch

#### What are graph networks?

A graph network takes a graph as input and returns a graph as output. The input
graph has edge- (*E* ), node- (*V* ), and global-level (**u**) attributes. The
output graph has the same structure, but updated attributes. Graph networks are
part of the broader family of "graph neural networks" (Scarselli et al., 2009).

To learn more about graph networks, see our arXiv paper: [Relational inductive
biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261).

![Graph network](images/graph-network.png)



## Usage example

You can see a forward calculation in `demo.py`
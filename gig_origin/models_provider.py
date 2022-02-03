from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU, ReLU
from torch.nn import functional as F
from torch_geometric.nn import  GraphConv ,DynamicEdgeConv
from torch_geometric.nn.models.basic_gnn import GIN

from torch_geometric.nn.glob.glob import global_add_pool, global_mean_pool
from torch_geometric.nn import GlobalAttention, Set2Set
import copy
from common_utils import *
seed_everything(seed=1234)

from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

#MODEL UTILS

try:
    from torch_cluster import knn
except ImportError:
    knn = None

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class DynamicEdgeConv(MessagePassing):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Number of nearest neighbors.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(DynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`DynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)


    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        """"""
        initial_x = copy.copy(x)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `DynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None), edge_index, initial_x


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


def DGCNN_layer(in_size, out_size):
    DGCNN_conv = Sequential(Linear(2 * in_size, out_size), LeakyReLU())

    return DynamicEdgeConv(DGCNN_conv, k=10)  # 10 #change to fix graph!!!!

#MODEL DEFINITION
#Node-level module
class NodeConvolution(Module):
    def __init__(self, config):
        super(NodeConvolution, self).__init__()
        self.config = config
        self.conv_list = ModuleList()

        if self.config.node_level_module == "GIN":
            self.conv_list.append(GIN(in_channels=config.num_node_features,
                                      hidden_channels= config.arg.node_conv_size[0],
                                      num_layers=5,
                                      out_channels=config.arg.node_conv_size[-1]))
        elif self.config.node_level_module == "GraphConv":





            self.conv_list.append(GraphConv(config.num_node_features, config.arg.node_conv_size[0]))
            for i in np.arange(1, len(config.arg.node_conv_size)):
                self.conv_list.append(GraphConv(config.arg.node_conv_size[i - 1], config.arg.node_conv_size[i]))


        gate_nn = Sequential(Linear(config.arg.node_conv_size[-1], int(config.arg.node_conv_size[-1]/2)), LeakyReLU(),
                             Linear(int(config.arg.node_conv_size[-1]/2), 1)
                   )
        # self.GlobalAttention_pooling = GlobalAttention(gate_nn=gate_nn)
        # self.Set2Set_pooling = Set2Set(config.arg.node_conv_size[-1], 50)
        if config.pooling == 'add':
            self.pooling = global_add_pool
        elif config.pooling == 'mean':
            self.pooling = global_mean_pool
        elif config.pooling == 'attention':
            self.pooling = GlobalAttention(gate_nn=gate_nn)
        elif config.pooling == 'set2set':
            self.pooling = Set2Set(config.arg.node_conv_size[-1], 50)
        else:
            print("This version is not implemented.")

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        if 'edge_weights' in data:
            x = self.conv_list[0](x, edge_index, data.edge_weights)
        elif 'edge_attr' in data:
            x = self.conv_list[0](x, edge_index, data.edge_attr[:,0])#torch.sum(data.edge_attr, dim=1))
        else:
            x = self.conv_list[0](x, edge_index)
        x = F.relu(x)

        for conv in self.conv_list[1:]:
            x = conv(x, edge_index)
            x = F.relu(x)  # F.leaky_relu(x)
            if self.config.use_dropout:
                x = F.dropout(x, p=self.config.dropout, training=self.training)

        # x = global_mean_pool(x, data.batch)  # proteins should be with mean pooling!
        # x = self.GlobalAttention_pooling(x, data.batch)
        # x = global_add_pool(x, data.batch) # for tox21

        # x = self.Set2Set_pooling(x, data.batch)
        x = self.pooling(x, data.batch)





        return x
# GraphInGraph model with population-level module
class GraphInGraph(Module):
    def __init__(self, config):
        super().__init__()
        #NODE-LEVEL MODULE
        if config.data_level == 'graph':
            self.node_conv = NodeConvolution(config)
        else:
            print("Not implemented")

        self.config = config

        # POPULATION-LEVEL MODULE
        self.conv_list = ModuleList()
        if config.pooling == 'set2set':
            input_size = config.arg.node_conv_size[-1]*2
        else:
            input_size = config.arg.node_conv_size[-1]

        #DGCNN
        if config.DGCNN:
            self.conv_list.append(DGCNN_layer(input_size, config.arg.dgcnn_conv_size[0]))
            for i in np.arange(1, len(config.arg.dgcnn_conv_size)):
                self.conv_list.append(DGCNN_layer(config.arg.dgcnn_conv_size[i - 1], config.arg.dgcnn_conv_size[i]))
            # GCN
            self.graph_conv = ModuleList()
            if len(config.arg.graph_conv_size)>0:
                self.graph_conv.append(GraphConv(config.arg.dgcnn_conv_size[-1], config.arg.graph_conv_size[0], aggr='mean'))
                for i in np.arange(1, len(config.arg.graph_conv_size)):
                    l = GraphConv(config.arg.graph_conv_size[i - 1], config.arg.graph_conv_size[i], aggr='mean')
                    self.graph_conv.append(l)
        #LGL
        else:
            self.conv_list.append(torch.nn.Linear(input_size, config.arg.graph_lin_size[0]))

            for i in np.arange(1, len(config.arg.graph_lin_size)):
                l = torch.nn.Linear(config.arg.graph_lin_size[i - 1], config.arg.graph_lin_size[i])
                self.conv_list.append(l)

            self.weight_layer = torch.nn.Linear(input_size, config.arg.graph_lin_size[-1])

            self.activation = ReLU()

            self.temp = torch.nn.Parameter(torch.tensor(config.temp_initial, requires_grad=True, device=self.config.device))
            self.theta = torch.nn.Parameter(torch.tensor(config.theta_initial, requires_grad=True, device=self.config.device))

            if self.config.target_distribution == 'power_law':
                self.mu = torch.nn.Parameter(torch.tensor(-1.5, requires_grad=True, device=self.config.device))
                self.sigma = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device=self.config.device))
            else:
                self.mu = torch.nn.Parameter(torch.tensor(9.0, requires_grad=True, device=self.config.device))#9
                self.sigma = torch.nn.Parameter(torch.tensor(3.0, requires_grad=True, device=self.config.device))#

            # before classification to obtain the graph representation
            # GCN
            self.graph_conv = ModuleList()
            if len(config.arg.graph_conv_size)>0:
                self.graph_conv.append(GraphConv(config.arg.graph_lin_size[-1], config.arg.graph_conv_size[0],aggr='mean'))
                for i in np.arange(1, len(config.arg.graph_conv_size)):
                    l = GraphConv(config.arg.graph_conv_size[i - 1], config.arg.graph_conv_size[i], aggr='mean')
                    self.graph_conv.append(l)


        # define final fc layers for classification
        if config.output_dim == 2:
            output_dim = config.output_dim - 1
        elif config.dataset == 'Tox21':
            output_dim = config.output_dim
        else:
            output_dim = config.output_dim

        # CLASSIFIER
        if len(config.arg.classif_fc) > 0:
            if len(config.arg.graph_conv_size) > 0:
                fc_list = [Linear(config.arg.graph_conv_size[-1], config.arg.classif_fc[0])]
            else:
                if config.DGCNN:
                    fc_list = [Linear(config.arg.dgcnn_conv_size[-1], config.arg.classif_fc[0])]
                else:
                    fc_list = [Linear(config.arg.graph_lin_size[-1], config.arg.classif_fc[0])]
            for i in np.arange(1, len(config.arg.classif_fc)):
                fc_list.append(ReLU())  # LeakyReLU()
                fc_list.append(Linear(config.arg.classif_fc[i - 1], config.arg.classif_fc[i]))
            fc_list.append(ReLU())
            fc_list.append(Linear(config.arg.classif_fc[- 1], output_dim))
        else:

            fc_list = [Linear(config.arg.graph_conv_size[-1], output_dim)]

        self.fc = Sequential(*fc_list)


    def forward(self, data):
            x = self.node_conv(data)

            if self.config.DGCNN:
                for conv in self.conv_list:
                    x, dgcnn_edge_index, dgcnn_x = conv(x)
                    x = F.relu(x)
                    if self.config.use_dropout:
                        x = F.dropout(x,p=self.config.dropout, training=self.training)

            else:
                for k, model in enumerate(self.conv_list):
                    if k == 0:
                        out_x = model(x)
                        out_x = self.activation(out_x)
                    else:
                        out_x = model(out_x)
                        out_x = self.activation(out_x)


                x = self.weight_layer(x)
                # compute pairwise distance
                diff = out_x.unsqueeze(1) - out_x.unsqueeze(0)
                # compute the norm
                diff = torch.pow(diff, 2).sum(2)
                mask_diff = diff != 0.0
                dist = - torch.sqrt(diff + torch.finfo(torch.float32).eps)
                dist = dist*mask_diff
                prob_matrix = self.temp * dist + self.theta

                adj = prob_matrix + torch.eye(prob_matrix.shape[0]).to(self.config.device)
                adj = torch.sigmoid(adj)


                diag_inv = torch.diag(adj.sum(-1)).inverse()
                x = torch.mm(diag_inv, torch.mm(adj, x))
                x = self.activation(x)

                edge_index = (adj > 0).nonzero().t()
                row, col = edge_index
                edge_weight = adj[row, col]
                g_x = copy.copy(x)

            if self.config.DGCNN:
                for conv in self.graph_conv:
                    x = conv(x, dgcnn_edge_index)
                    x = ReLU()(x)
                    if self.config.use_dropout:
                        x = F.dropout(x, p=self.config.dropout, training=self.training)
            else:
                if len(self.graph_conv)>0:
                    x = self.graph_conv[0](x, edge_index
                             , edge_weight
                             )
                    x = ReLU()(x)
                    for conv in self.graph_conv[1:]:
                        x = conv(x, edge_index)
                        x = ReLU()(x)
                        if self.config.use_dropout:
                            x = F.dropout(x,p=self.config.dropout, training=self.training)

            x = self.fc(x)
            if self.config.output_dim == 2:
                x = x.view(-1)

            if self.config.DGCNN:
                return x, dgcnn_edge_index, dgcnn_x

            # compute kl loss for 2 normals, learnable
            if self.config.target_distribution == 'power_law':
                target_distr = torch.zeros(self.config.batch_size_train).to(self.config.device)
                target_distr[4:] = self.sigma*(1.0 + torch.arange(self.config.batch_size_train - 4).to(self.config.device)).pow(self.mu).to(self.config.device)
            else:
                target_distr = torch.zeros(self.config.batch_size_train).to(self.config.device)
                target_distr[4:] = torch.exp(-(self.mu - torch.arange(self.config.batch_size_train-4).to(self.config.device) + 1.0) ** 2 / (self.sigma ** 2)).to(
                    self.config.device)

            target_distr = target_distr / target_distr.sum()

            return x, g_x, edge_index, edge_weight, adj, target_distr

#Baseline models
class FixedGraphInGraph(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.data_level == 'graph':
            self.node_conv = NodeConvolution(config)
        else:
            print("Not implemented")


        self.graph_conv = ModuleList()
        self.graph_conv.append(GraphConv(config.arg.node_conv_size[-1], config.arg.graph_conv_size[0]))
        for i in np.arange(1, len(config.arg.graph_conv_size)):
            l = GraphConv(config.arg.graph_conv_size[i - 1], config.arg.graph_conv_size[i])
            self.graph_conv.append(l)

        if config.output_dim == 2:
            output_dim = config.output_dim - 1
        else:
            output_dim = config.output_dim


        if len(config.arg.classif_fc) > 0:
            fc_list = [Linear(config.arg.graph_conv_size[-1], config.arg.classif_fc[0])]
            for i in np.arange(1, len(config.arg.classif_fc)):
                fc_list.append(ReLU())
                fc_list.append(Linear(config.arg.classif_fc[i - 1], config.arg.classif_fc[i]))
            fc_list.append(Linear(config.arg.classif_fc[- 1], output_dim))
        else:
            fc_list = [Linear(config.arg.graph_conv_size[-1], output_dim)]

        self.fc = Sequential(*fc_list)


    def forward(self, data, edge_index):
            x = self.node_conv(data)

            for conv in self.graph_conv:
                x = conv(x, edge_index)
                x = ReLU()(x)
                if self.config.use_dropout:
                    x = F.dropout(x,p=self.config.dropout, training=self.training)

            x = self.fc(x)
            if self.config.output_dim == 2:
                x = torch.sigmoid(x.view(-1))


            return x
class GCN(Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.config = config
        # define final fc layers for classification
        if config.output_dim == 2:
            self.output_dim = config.output_dim - 1
        elif config.dataset == 'Tox21':
            self.output_dim = config.output_dim
        else:
            self.output_dim = config.output_dim

        self.conv_list = ModuleList()
        self.lin_list = ModuleList()

        if config.DGCNN:
            self.conv_list.append(DGCNN_layer(76, config.arg.dgcnn_conv_size[0]))
            for i in np.arange(1, len(config.arg.dgcnn_conv_size)):
                self.conv_list.append(DGCNN_layer(config.arg.dgcnn_conv_size[i - 1], config.arg.dgcnn_conv_size[i]))
            self.conv_list.append(DGCNN_layer(config.arg.dgcnn_conv_size[-1], config.arg.classif_fc[0]))

        else:
            self.conv_list.append(GraphConv(config.num_node_features, config.arg.node_conv_size[0]))
            for i in np.arange(1, len(config.arg.node_conv_size)):
                self.conv_list.append(GraphConv(config.arg.node_conv_size[i - 1], config.arg.node_conv_size[i]))
            self.conv_list.append(GraphConv(config.arg.node_conv_size[-1], config.arg.classif_fc[0]))



        for i in np.arange(1, len(config.arg.classif_fc)):
            self.lin_list.append(Linear(config.arg.classif_fc[i - 1], config.arg.classif_fc[i]))

        self.lin_list.append(Linear(config.arg.classif_fc[-1],self.output_dim))

        if config.pooling == 'add':
            self.pooling = global_add_pool
        elif config.pooling == 'mean':
            self.pooling = global_mean_pool
        else:
            print("This version is not implemented.")




    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        if self.config.DGCNN:
            for conv in self.conv_list:
                x, dgcnn_edge_index, dgcnn_x = conv(x.view(100,-1))
                x = F.relu(x)
                if self.config.use_dropout:
                    x = F.dropout(x, p=self.config.dropout, training=self.training)
        else:
            if 'edge_weights' in data:
                x = self.conv_list[0](x, edge_index, data.edge_weights)
            elif 'edge_attr' in data:
                x = self.conv_list[0](x, edge_index, data.edge_attr[:, 0])
            else:
                x = self.conv_list[0](x, edge_index)
            x = F.relu(x)
            for conv in self.conv_list[1:]:
                x = conv(x, edge_index)
                x = F.relu(x)
                if self.config.use_dropout:
                    x = F.dropout(x, p=self.config.dropout, training=self.training)


        for lin in self.lin_list[:-1]:
            x = lin(x)
            x = F.relu(x)
            if self.config.use_dropout:
                x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = self.lin_list[-1](x)
        if not self.config.DGCNN:
            x = self.pooling(x, data.batch)
        if self.config.output_dim == 2:
            x = x.view(-1)




        if self.config.DGCNN:
            return x, dgcnn_edge_index, dgcnn_x
        else:
            return x

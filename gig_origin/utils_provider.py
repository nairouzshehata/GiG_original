import pandas as pd
from pyvis.network import Network
from skimage.segmentation import slic, mark_boundaries
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset, MoleculeNet
from torch.utils.data import random_split
from torch_geometric.data import Data
import glob
import nibabel
from sklearn.preprocessing import normalize, RobustScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets
from models_provider import GraphInGraph,  FixedGraphInGraph, GCN
import mlflow
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import dense_to_sparse, erdos_renyi_graph, to_dense_adj, subgraph, to_dense_batch
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from common_utils import *
from datasets import *

seed_everything(seed=1234)

DATASETS2YAML = {"REDDIT-BINARY": "reddit_binary.yaml",
                               "REDDIT-MULTI-5K": "reddit_multi_5k.yaml",
                               "IMDB-BINARY": "imdb_binary.yaml",
                               "IMDB-MULTI": "imdb_multi.yaml",
                               "COLLAB": "collab.yaml",
                               "MUTAG": "mutag.yaml",
                                "ENZYMES": "enzymes.yaml",
                 "DD":"dd.yaml",
                 "NCI1":"nci1.yaml",
                               "HCP": "hcp.yaml",
                               "PROTEINS_full": "proteins.yaml",
                               "Tox21": "tox21.yaml"}

DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD
}
def define_dataset(dataset_name, data_dir):

    def preprocess_dataset(name, args_dict):
        dataset_class = DATASETS[name]
        if name == 'ENZYMES':
            args_dict.update(use_node_attrs=True)
        return dataset_class(**args_dict)

    # if __name__ == "__main__":
    # args_dict = {'DATA_DIR': 'DATA/CHEMICAL',
    #              # 'dataset_name': 'PROTEINS',
    #              'outer_k': 10,
    #              'inner_k': None,
    #              'use_one': False,
    #              'use_node_degree': False,
    #              'precompute_kron_indices': True}

    args_dict = {'DATA_DIR': data_dir,
                 # 'dataset_name': 'PROTEINS',
                 'outer_k': 10,
                 'inner_k': None,
                 'use_one': True,
                 'use_node_degree': False,
                 'precompute_kron_indices': True}


    dataset = preprocess_dataset(dataset_name, args_dict)
    return dataset

# code from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0,
                 path='checkpoints/checkpoint.pt', trace_func=print, config=None, continue_training=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.config = config
        self.continue_training = continue_training

    def __call__(self, epoch, val_loss, model, l_plot):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if not self.continue_training:
                self.save_checkpoint(epoch, val_loss, model, l_plot)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, l_plot)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, l_plot):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({'epoch': epoch + 1, 'state_dict':
            model.state_dict()}, self.path + "/checkpoint")

        # if not self.config.DGCNN:
        torch.save(l_plot, self.path + '/epoch_last' + '_l_plot.pt')

        self.val_loss_min = val_loss

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma, device='cuda'):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers.to(device)

    def forward(self, x):
        d = torch.cdist(self.centers[:, None], x[:, None])
        x = torch.softmax(-d ** 2 / self.sigma ** 2, dim=0)  # - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x


def kl_div(p, q):
    return torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8))


def compute_distr(adj, softhist):
    deg = adj.sum(-1)
    distr = softhist(deg)
    return distr / torch.sum(distr), deg


# Baselines. Creation of fixed population graphs.
def create_knn_proteins(dataloader, config, k=10):
    graph_list_loader = []
    gk = WeisfeilerLehman(normalize=True, base_graph_kernel=VertexHistogram)
    for data in dataloader:
        graph_list = []
        for i in range(torch.max(data.batch) + 1):
            graph_idx = (data.batch == i).nonzero().view(-1)
            graph_edge_index, _ = subgraph(subset=graph_idx, edge_index=data.edge_index)
            # not the best solution for graphs without edges!
            if graph_edge_index.shape[1] == 0:
                graph_edge_index = torch.Tensor([[[item], [item]] for item in graph_idx]).reshape(2, -1)
                graph_edge_index = graph_edge_index.int()

            max_value, min_value = torch.max(graph_edge_index), torch.min(graph_edge_index)

            graph_edge_index = [(int(graph_edge_index[0][i]), int(graph_edge_index[1][i])) for i in
                                range(graph_edge_index.shape[1])]


            graph_node_labels = {i: int(data.x[i].argmax(-1)) for i in range(min_value, max_value + 1)}
            if config.dataset in ['Tox21']:
                graph_node_labels = {i: int("".join(map(str, [int(el) for el in data.x[i]]))) for i in range(min_value, max_value + 1)}
            if config.dataset in ['HCP']:
                graph_node_labels = {i: int(float("".join(map(str, [int(abs(el)) for el in data.x[i]])))) for i in range(min_value, max_value + 1)}
            graph_list.append(Graph(graph_edge_index, node_labels=graph_node_labels))

        K = gk.fit_transform(graph_list)
        population_graph_edge_indexes = torch.tensor(
            [(n, el) for n in range(K.shape[0]) for el in np.argsort(K[n])[-k:]])
        edge_indexes = torch.zeros(2, population_graph_edge_indexes.shape[0]).long()
        edge_indexes[0] = population_graph_edge_indexes[:, 0]
        edge_indexes[1] = population_graph_edge_indexes[:, 1]
        graph_list_loader.append(torch.tensor(edge_indexes).to(config.device))
    return graph_list_loader


def create_random(dataloader, config, p=0.1):
    edge_index_loader = []
    for data in dataloader:
        batch_size = torch.max(data.batch) + 1
        edge_index = erdos_renyi_graph(batch_size, p)
        edge_index_loader.append(edge_index.to(config.device))
    return edge_index_loader

def create_dataset_loaders(config, shuffle_train = True):
    if config.dataset == 'mnist':
        if config.data_level == 'graph' or config.data_level == 'feature_matrix':
            pg_train_dataset = torch.load("data/" + str(config.k) + "train_pg_graphs")
            pg_test_dataset = torch.load("data/" + str(config.k) + "test_pg_graphs")

            num_training = int(len(pg_train_dataset) * 0.9)
            num_val = len(pg_train_dataset) - (num_training)

            training_set, validation_set = random_split(pg_train_dataset, [num_training, num_val],
                                                        generator=torch.Generator().manual_seed(42))

            loader_train = DataLoader(training_set, batch_size=config.batch_size_train, shuffle=shuffle_train)
            loader_val = DataLoader(validation_set, batch_size=config.batch_size_val, shuffle=False)

            loader_test = DataLoader(pg_test_dataset, batch_size=config.batch_size_test, shuffle=False)  # was False
            config.num_node_features = pg_train_dataset[0].num_node_features
            config.num_segments = pg_train_dataset[0].x.shape[0]

        else:
            print("This version of MNIST dataset is not implemented")
    elif config.dataset in ["PROTEINS_full"]:
        dataset = TUDataset(os.path.join('data', config.dataset), name=config.dataset, use_node_attr=True)


        config.output_dim = dataset.num_classes
        config.input_dim = dataset.num_features
        config.num_node_features = dataset.num_features
        # TODO: change, just fast version
        # if config.dataset in ["PROTEINS_full"]:
        #     dataset_proteins = define_dataset(dataset_name=config.dataset, data_dir='data/CHEMICAL')
        #     dataset_proteins = dataset_proteins.dataset.get_data()
        #     dataset_list = []
        #     for i, data in enumerate(dataset):
        #         data.x = dataset_proteins[i].x
        #         dataset_list.append(data)
        #     dataset = dataset_list
        #     config.input_dim = 3
        #     config.num_node_features = 3


        num_training = int(len(dataset) * 0.9)
        num_test = len(dataset) - (num_training)
        training_set, test_set = random_split(dataset, [num_training, num_test],
                                              generator=torch.Generator().manual_seed(42))
        num_training = int(len(training_set) * 0.9)
        num_val = len(training_set) - (num_training)

        training_set, validation_set = random_split(training_set, [num_training, num_val],
                                                    generator=torch.Generator().manual_seed(42))

        loader_train = DataLoader(training_set, batch_size=config.batch_size_train, shuffle=shuffle_train)
        loader_val = DataLoader(validation_set, batch_size=config.batch_size_val, shuffle=False)
        loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)


    elif config.dataset in ['Tox21']:

        dataset = PygGraphPropPredDataset(name="ogbg-moltox21", root='data/')
        if config.label_idx is not None:
            config.output_dim = 2
        elif config.label_treshold is not None:
            config.output_dim = 12
        else:
            config.output_dim = 12
        config.input_dim = dataset.num_features
        config.num_node_features = dataset.num_features
        split_idx = dataset.get_idx_split()
        loader_train = DataLoader(dataset[split_idx["train"]], batch_size=config.batch_size_train, shuffle=shuffle_train)
        loader_val = DataLoader(dataset[split_idx["valid"]], batch_size=config.batch_size_val, shuffle=False)
        loader_test = DataLoader(dataset[split_idx["test"]], batch_size=config.batch_size_test, shuffle=False)

    elif config.dataset in ['HCP']:
        path_root = os.path.join('data', 'HCP_PTN1200')
        path = os.path.join(path_root, 'netmats/3T_HCP1200_MSMAll_d' + str(config.hcp_dim) + '_ts2')
        filepath_feature_matrix = os.path.join(path, 'netmats1.txt')
        filepath_adjacency_matrix = os.path.join(path, 'Mnet1.pconn.nii')
        labels_path = os.path.join(path_root, 'labels_gender_age.npy')
        dataset = create_hcp_dataloader(filepath_feature_matrix, filepath_adjacency_matrix, labels_path, config.task,
                                        dim=config.hcp_dim)
        config.input_dim = dataset[0].num_features
        config.num_node_features = dataset[0].num_node_features
        if config.task == 'age':

            config.output_dim = 4

            num_training = int(len(dataset) * 0.85)
            num_test = len(dataset) - (num_training)
            training_set, test_set = random_split(dataset, [num_training, num_test],
                                                  generator=torch.Generator().manual_seed(42))
            num_training = int(len(training_set) * 0.8)
            num_val = len(training_set) - (num_training)

            training_set, validation_set = random_split(training_set, [num_training, num_val],
                                                        generator=torch.Generator().manual_seed(42))
            if shuffle_train:
                weights_train = {0: 1, 1:1, 2:1, 3:0.001}
                weights_val = {0: 1, 1:1, 2:1, 3:0.001}

                weights_train = torch.Tensor([weights_train[int(data.y)] for data in training_set])
                weights_val = torch.Tensor([weights_val[int(data.y)] for data in validation_set])
                sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train,num_samples=round(2*len(training_set)),
                                                                       replacement=True,generator=torch.Generator().manual_seed(42))
                sampler_val = torch.utils.data.WeightedRandomSampler(weights=weights_val,num_samples=round(len(validation_set)),
                                                                     replacement=True,generator=torch.Generator().manual_seed(42))

                loader_train = DataLoader(training_set, batch_size=config.batch_size_train,sampler=sampler_train)
                loader_val = DataLoader(validation_set, batch_size=config.batch_size_test, sampler=sampler_val)
                loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)
            else:
                num_training = int(len(dataset) * 0.72)
                num_val = int(len(dataset) * 0.08)

                num_test = len(dataset) - (num_training) - (num_val)

                training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test],
                                                                      generator=torch.Generator().manual_seed(42))

                loader_train = DataLoader(training_set, batch_size=config.batch_size_train, shuffle=shuffle_train)
                loader_val = DataLoader(validation_set, batch_size=config.batch_size_test, shuffle=False)
                loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)

        else:
            config.output_dim = 2
            num_training = int(len(dataset) * 0.72)
            num_val = int(len(dataset) * 0.08)

            num_test = len(dataset) - (num_training) - (num_val)

            training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test],
                                                        generator=torch.Generator().manual_seed(42))

            loader_train = DataLoader(training_set, batch_size=config.batch_size_train, shuffle=shuffle_train)
            loader_val = DataLoader(validation_set, batch_size=config.batch_size_test, shuffle=False)
            loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)


    else:
        print("Dataset is not implemented")
    # version with dynamic populated graph
    return loader_train, loader_val, loader_test


def create_model(config, loader_train, loader_val, loader_test):
    # model
    fixed_train_graphs, fixed_test_graphs, fixed_val_graphs = [], [], []

    if config.model == 'FixedGraphInGraph':
        model = FixedGraphInGraph(config)
        # create fix population level graphs
        if config.fix_population_graph == 'random':
            random_saved_files_path = config.arg.saving_path
            if os.path.exists(random_saved_files_path + "/l_fixed_train_graphs.pt"):
                fixed_train_graphs = torch.load(random_saved_files_path + '/l_fixed_train_graphs.pt')
                fixed_test_graphs = torch.load(random_saved_files_path + '/l_fixed_test_graphs.pt')
                fixed_val_graphs = torch.load(random_saved_files_path + '/l_fixed_val_graphs.pt')
            else:
                fixed_train_graphs = create_random(loader_train, config)
                fixed_val_graphs = create_random(loader_val, config)
                fixed_test_graphs = create_random(loader_test, config)
                torch.save(fixed_train_graphs, random_saved_files_path + '/l_fixed_train_graphs.pt')
                torch.save(fixed_val_graphs, random_saved_files_path + '/l_fixed_val_graphs.pt')
                torch.save(fixed_test_graphs, random_saved_files_path + '/l_fixed_test_graphs.pt')

        elif config.fix_population_graph == 'knn':
            if os.path.exists(config.arg.saving_path + "/fixed_train_graphs.pt"):
                fixed_train_graphs = torch.load(config.arg.saving_path + '/fixed_train_graphs.pt')
                fixed_val_graphs = torch.load(config.arg.saving_path + '/fixed_val_graphs.pt')
                fixed_test_graphs = torch.load(config.arg.saving_path + '/fixed_test_graphs.pt')
            else:

                if config.dataset == 'PROTEINS' or config.dataset == "PROTEINS_full":
                    dataset = TUDataset(os.path.join('data', config.dataset),
                                                name=config.dataset, use_node_attr=False)

                    num_training = int(len(dataset) * 0.9)
                    num_test = len(dataset) - (num_training)
                    training_set, test_set = random_split(dataset, [num_training, num_test],
                                                          generator=torch.Generator().manual_seed(42))
                    num_training = int(len(training_set) * 0.9)
                    num_val = len(training_set) - (num_training)

                    training_set, validation_set = random_split(training_set, [num_training, num_val],
                                                                generator=torch.Generator().manual_seed(42))

                    loader_train = DataLoader(training_set, batch_size=config.batch_size_train, shuffle=False)
                    loader_val = DataLoader(validation_set, batch_size=config.batch_size_val, shuffle=False)
                    loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)

                    fixed_train_graphs = create_knn_proteins(loader_train, config)
                    fixed_val_graphs = create_knn_proteins(loader_val, config)
                    fixed_test_graphs = create_knn_proteins(loader_test, config)
                elif config.dataset in ['Tox21']:
                    fixed_train_graphs = create_knn_proteins(loader_train, config)
                    fixed_val_graphs = create_knn_proteins(loader_val, config)
                    fixed_test_graphs = create_knn_proteins(loader_test, config)
                elif config.dataset in ['HCP']:
                    path_root = os.path.join('data', 'HCP_PTN1200')
                    path = os.path.join(path_root, 'netmats/3T_HCP1200_MSMAll_d' + str(15) + '_ts2')
                    filepath_feature_matrix = os.path.join(path, 'netmats1.txt')
                    filepath_adjacency_matrix = os.path.join(path, 'Mnet1.pconn.nii')
                    labels_path = os.path.join(path_root, 'labels_gender_age.npy')
                    dataset = create_hcp_dataloader(filepath_feature_matrix, filepath_adjacency_matrix, labels_path,
                                                    config.task,
                                                    dim=15)
                    config.input_dim = dataset[0].num_features
                    config.num_node_features = dataset[0].num_node_features
                    if config.task == 'age':
                        config.output_dim = 4
                    else:
                        config.output_dim = 2  # gender
                    num_training = int(len(dataset) * 0.72)
                    num_val = int(len(dataset) * 0.08)

                    num_test = len(dataset) - (num_training) - (num_val)

                    training_set, validation_set, test_set = random_split(dataset,
                                                                          [num_training, num_val, num_test],
                                                                          generator=torch.Generator().manual_seed(
                                                                              42))


                    loader_train = DataLoader(training_set, batch_size=config.batch_size_train, shuffle=False)
                    loader_val = DataLoader(validation_set, batch_size=config.batch_size_test, shuffle=False)
                    loader_test = DataLoader(test_set, batch_size=config.batch_size_test, shuffle=False)




                    fixed_train_graphs = create_knn_proteins(loader_train, config)
                    fixed_val_graphs = create_knn_proteins(loader_val, config)
                    fixed_test_graphs = create_knn_proteins(loader_test, config)
                else:
                    print("Dataset is not implemented")


        else:
            print('Not implemented')

    else:
        model = GraphInGraph(config)
    model = model.to(config.device)
    # model.apply(weights_init)
    return model, fixed_train_graphs,fixed_val_graphs, fixed_test_graphs


def define_loss(config):
    if config.dataset in ['mnist', 'ogbg-ppa', 'ENZYMES']:
        criterion = nn.CrossEntropyLoss()
    elif config.dataset in ['Tox21']:
        criterion = nn.BCEWithLogitsLoss(pos_weight =torch.Tensor([13]).to(config.device)) #15


    elif config.dataset in ['PROTEINS', 'PROTEINS_full', 'HIV', 'BBBP', 'DD', 'PPI', 'NCI1']:
        criterion = nn.BCEWithLogitsLoss()
    elif config.dataset in ['HCP']:
        if config.task == 'gender':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

    else:
        print("This dataset is not implemented.")
    return criterion


def compute_losses(config, data, prediction_probs, criterion, adj=None, pytorch_target_distr=None, target_dist=None, softhist=None):
    kl_loss = torch.tensor(0)
    if config.output_dim == 2 and config.dataset not in ['Tox21', 'PPI']:
        ce_loss = criterion(prediction_probs, data.y.float())

    elif config.dataset == 'Tox21':
        if config.label_idx is not None:
            nan_mask = ~torch.isnan(data.y[:, config.label_idx])
            labels_el = data.y[:, config.label_idx][nan_mask]
            pred_el = prediction_probs[nan_mask].view(-1)

        elif config.label_treshold is not None:
            nan_mask = ~torch.isnan(data.y)
            labels_el = data.y[nan_mask]

            pred_el = prediction_probs[nan_mask]
        else:
            nan_mask = ~torch.isnan(data.y)
            labels_el = data.y[nan_mask]

            pred_el = prediction_probs[nan_mask]
        ce_loss = criterion(pred_el, labels_el)
    else:
        ce_loss = criterion(prediction_probs, data.y.long())

    if not config.DGCNN and not config.model == 'FixedGraphInGraph':
        if config.regularization == 'KL_degree_dist' and config.model !='GCN':


            binarized_adj = torch.zeros(adj.shape).to(config.device)
            binarized_adj[adj > 0.5] = 1# 0.5
            dist, deg = compute_distr(adj * binarized_adj, softhist)
            if config.learnable_distr:
                kl_loss = kl_div(dist, pytorch_target_distr)
            else:
                kl_loss = kl_div(dist, target_dist)

    return ce_loss, kl_loss


def compute_metrics(data, prediction_probs, adj=None, config=None):
    labels = data.y
    if config.output_dim == 2 or config.dataset == 'Tox21':
        pred = torch.round(torch.sigmoid(prediction_probs))
    else:
        pred = torch.argmax(prediction_probs, dim=1)  # should be here added softmax?

    if config.dataset == 'Tox21':

        if config.label_idx is not None:
            nan_mask = ~torch.isnan(labels[:, config.label_idx])
            labels_el = labels[:, config.label_idx][nan_mask]
            pred_el = pred[nan_mask].view(-1)
            prediction_probs_ = prediction_probs[nan_mask].view(-1)
        elif config.label_treshold is not None:
            nan_mask = ~torch.isnan(labels[:, config.label_treshold])
            labels_el = labels[:, config.label_treshold][nan_mask]
            pred_el = pred[:, config.label_treshold][nan_mask].view(-1)
            prediction_probs_ = prediction_probs[:, config.label_treshold][nan_mask].view(-1)
        else:
            nan_mask = ~torch.isnan(labels)
            labels_el = labels[nan_mask]
            pred_el = pred[nan_mask]
            prediction_probs_ = prediction_probs[nan_mask]
        acc = accuracy_score(labels_el.cpu().detach().numpy(), pred_el.cpu().detach().numpy())
        # FIX smth with this metric
        try:
            roc_auc_score_value = roc_auc_score(labels_el.cpu().detach().numpy(),
                                                prediction_probs_.cpu().detach().numpy())
        except:
            roc_auc_score_value = np.nan

    else:
        acc = accuracy_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())

        try:

            roc_auc_score_value = roc_auc_score(labels.cpu().detach().numpy(),
                                                prediction_probs.cpu().detach().numpy())
        except:
            roc_auc_score_value = np.nan

    return acc, roc_auc_score_value


def create_hcp_dataloader(filepath_feature_matrix, filepath_adjacency_matrix, labels_path, task='gender', dim=200):
    labels = np.load(labels_path)

    feature_matrix = torch.Tensor(np.loadtxt(filepath_feature_matrix))
    feature_matrix = feature_matrix.view(-1, dim, dim)
    loaded_adj = nibabel.load(filepath_adjacency_matrix)
    adjacency_matrix = torch.Tensor(normalize(loaded_adj.get_fdata()))
    # adjacency_matrix = torch.Tensor(loaded_adj.get_fdata())

    edge_index, edge_attributes = dense_to_sparse(adjacency_matrix)
    if task == 'age':
        y_j = 2
    else:
        y_j = 1
    # data_list = [Data(x=torch.Tensor(RobustScaler().fit_transform(feature_matrix[i])).float(), edge_index=edge_index.long(),
    #                   y=torch.Tensor([labels[i][y_j]]), edge_weights=edge_attributes) for i in range(feature_matrix.shape[0])]
    data_list = [
        Data(x=torch.Tensor(feature_matrix[i]).float(), edge_index=edge_index.long(),
             y=torch.Tensor([labels[i][y_j]]), y_age=torch.Tensor([labels[i][2]]), y_gender=torch.Tensor([labels[i][1]]),  edge_weights=edge_attributes) for i in range(feature_matrix.shape[0])]
    return data_list


def train_model(config, loader_train, loader_val, loader_test):
    mlflow.log_param('dataset', config.dataset)
    mlflow.log_param('target_distribution', config.target_distribution)
    mlflow.log_param('target_distribution_learnable', config.learnable_distr)
    mlflow.log_param('lr', config.lr)
    mlflow.log_param('fix_population_level', config.fix_population_level)
    model, fixed_train_graphs, fixed_val_graphs, fixed_test_graphs = create_model(config, loader_train, loader_val, loader_test)
    print("Current model", model)

    # optimizer
    op_par = [param for name_, param in model.named_parameters() if name_ not in ['temp', 'theta']]
    optimizer = torch.optim.Adam(op_par, lr=config.lr
                                 # , weight_decay=1e-3#, amsgrad=True  # , weight_decay=0 -4
                                 )

    if config.model != 'FixedGraphInGraph' and not config.DGCNN:
        optimizer_temp = torch.optim.Adam([model.temp], lr=config.temp_lr
                                          # , weight_decay=1e-3#, amsgrad=True  # , weight_decay=0 -4
                                          )
        optimizer_theta = torch.optim.Adam([model.theta], lr=config.theta_lr
                                           # , weight_decay=1e-3#, amsgrad=True  # , weight_decay=0 -4
                                           )
        optimizer_sigma = torch.optim.Adam([model.sigma], lr=config.sigma_lr#config.lr
                                           # , weight_decay=1e-3#, amsgrad=True  # , weight_decay=0 -4
                                           )
        optimizer_mu = torch.optim.Adam([model.mu], lr=config.mu_lr#config.lr
                                        # , weight_decay=1e-3#, amsgrad=True  # , weight_decay=0 -4
                                        )

        if config.dataset in ['Tox21', 'HCP', 'mnist']:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
                                          threshold_mode='abs')
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=10)

    else:
        if config.dataset in ['Tox21', 'HCP', 'mnist']:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
                                      threshold_mode='abs')
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=10)


    writer = SummaryWriter(config.arg.writer_path)

    criterion = define_loss(config)

    # Load trained model from path=config.arg.saving_path
    initial_epoch = 0
    continue_training = False
    if os.path.exists(config.arg.saving_path + "/checkpoint"):
        checkpoint = torch.load(config.arg.saving_path + "/checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        initial_epoch = checkpoint['epoch']
        continue_training = True

    early_stopping = EarlyStopping(patience=config.patience, verbose=True,
                                   path=config.arg.saving_path, config=config, continue_training=continue_training)

    n_nodes = config.batch_size_train
    softhist = SoftHistogram(bins=n_nodes, min=0.5, max=n_nodes + 0.5, sigma=0.6, device=config.device)

    if config.target_distribution == 'power_law':
        target_dist = torch.zeros(n_nodes).to(config.device)
        target_dist[4:] = (1 + torch.arange(n_nodes - 4)).pow(-1.5).to(config.device)
    else:
        target_dist = torch.exp(-(7. - torch.arange(n_nodes) + 1) ** 2 / 3 ** 2).to(config.device)  # gaussian lr=1e-4
        target_dist[:4] = 0.0

    kl_sum_par = config.kl_sum_par
    ce_sum_par = config.ce_sum_par

    mlflow.log_param('kl_sum_par', config.kl_sum_par)
    mlflow.log_param("batch_size_train", config.batch_size_train)

    target_dist = target_dist / target_dist.sum()

    # run training
    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    epoch_kl_loss_train = []
    epoch_kl_loss_test = []
    epoch_add_loss_train = []
    epoch_add_loss_test = []
    epoch_ce_loss_train = []
    epoch_ce_loss_test = []


    if config.regularization == 'KL_degree_dist':
        add_ce_loss_flag = False
    for epoch in range(initial_epoch, config.n_epoch):
        l_plot = []

        t = time.time()

        loss_train = []
        acc_train = []
        loss_test = []
        acc_test = []

        kl_loss_train = []
        kl_loss_test = []
        ce_loss_train = []
        ce_loss_test = []


        roc_auc_score_test = []
        roc_auc_score_value_train = []

        if config.data_level == 'graph':
            model.train()
            for j, data in enumerate(loader_train):
                train_loss = 0.0
                optimizer.zero_grad()
                if config.model != 'FixedGraphInGraph' and not config.DGCNN:
                    optimizer_theta.zero_grad()
                    optimizer_temp.zero_grad()
                    optimizer_mu.zero_grad()
                    optimizer_sigma.zero_grad()
                data = data.to(config.device)

                if config.DGCNN:
                    prediction_probs, dgcnn_edge_index, dgcnn_x = model(data)
                    # prediction_probs = model(data)

                    ce_loss, kl_loss = compute_losses(config, data, prediction_probs, criterion)
                elif config.model == 'FixedGraphInGraph':
                    prediction_probs = model(data, fixed_train_graphs[j])
                    ce_loss, kl_loss = compute_losses(config, data, prediction_probs, criterion)
                else:
                    prediction_probs, g_x, edge_index, edge_weight, adj, pytorch_target_distr = model(data)

                    ce_loss, kl_loss = compute_losses(config, data, prediction_probs, criterion,
                                                      adj, pytorch_target_distr, target_dist, softhist)

                ce_loss_train.append(ce_loss.item())
                kl_loss_train.append(kl_loss.item())
                train_loss += ce_sum_par*ce_loss
                train_loss += kl_sum_par * kl_loss  # for normal it was 0.3

                train_loss.backward()
                optimizer.step()
                if config.model != 'FixedGraphInGraph' and not config.DGCNN:
                    optimizer_temp.step()
                    optimizer_theta.step()
                    optimizer_mu.step()
                    optimizer_sigma.step()

                    acc, roc_auc_score_value = compute_metrics(data, prediction_probs, adj=adj, config=config)
                else:
                    acc, roc_auc_score_value = compute_metrics(data, prediction_probs, config=config)
                acc_train.append(acc)
                roc_auc_score_value_train.append(roc_auc_score_value)
                loss_train.append(train_loss.item())
                writer.add_scalar('Loss/train', train_loss.item())
                writer.add_scalar('Acc/train', acc)

            model.eval()

            for j, data in enumerate(loader_val):
                test_loss = 0.0
                data = data.to(config.device)
                if config.DGCNN:
                    prediction_probs, dgcnn_edge_index, dgcnn_x = model(data)
                    ce_loss, kl_loss = compute_losses(config, data, prediction_probs, criterion)

                elif config.model == 'FixedGraphInGraph':
                    prediction_probs = model(data, fixed_val_graphs[j])
                    ce_loss, kl_loss = compute_losses(config, data, prediction_probs, criterion)
                else:
                    # prediction_probs,g_x, edge_index, edge_weight, adj,kl_loss = model(data)
                    prediction_probs, g_x, edge_index, edge_weight, adj, pytorch_target_distr = model(data)
                    par_sigma = model.sigma

                    ce_loss, kl_loss = compute_losses(config, data, prediction_probs, criterion,
                                                      adj, pytorch_target_distr, target_dist, softhist)
                ce_loss_test.append(ce_loss.item())
                kl_loss_test.append(kl_loss.item())
                test_loss += ce_sum_par*ce_loss
                test_loss += kl_sum_par * kl_loss  # for normal it was 0.3

                if config.model == '' and not config.DGCNN:
                    acc, roc_auc_score_value = compute_metrics(data, prediction_probs, adj=adj, config=config)
                else:
                    acc, roc_auc_score_value = compute_metrics(data, prediction_probs,  config=config)

                if kl_loss.item() < config.kl_treshold:
                    kl_sum_par = config.kl_sum_par_after_tresh


                acc_test.append(acc)



                roc_auc_score_test.append(roc_auc_score_value)
                loss_test.append(test_loss.item())

                writer.add_scalar('Loss/val', test_loss.item())
                writer.add_scalar('Acc/val', acc)
                if config.DGCNN:
                    l_plot.append((dgcnn_x, dgcnn_edge_index, data.y))
                elif config.model == 'FixedGraphInGraph' or config.DGCNN:
                    pass
                else:
                    l_plot.append((g_x, edge_index, edge_weight, data.y, adj))
        else:
            print("This version is not implemented.")

        acc_train_ = np.nanmean(np.array(acc_train))
        acc_test_ = np.nanmean(np.array(acc_test))
        if acc_test_ > 0.39:
            ce_sum_par = config.ce_sum_par_after_tresh
        print('%d] loss train: %.2e loss test: %.2e acc: %.2f  acc_val: %.2f  (%.2f s)' % (
            epoch, np.nanmean(loss_train), np.nanmean(loss_test), acc_train_, acc_test_, (time.time() - t)))
        print('%d] sum_loss: %.2e loss kl: %.2f  ce loss: %.2f  roc_auc_score_train: %.2f  roc_auc_score_test: %.2f (%.2f s)' % (
                  epoch, np.mean(loss_test),  np.mean(kl_loss_test), np.mean(ce_loss_test),
                  np.nanmean(roc_auc_score_value_train), np.nanmean(roc_auc_score_test), (time.time() - t)))

        epoch_loss_train.append(np.nanmean(loss_train))
        epoch_acc_train.append(acc_train_)
        epoch_loss_test.append(np.nanmean(loss_test))
        epoch_acc_test.append(acc_test_)

        epoch_kl_loss_train.append(np.nanmean(kl_loss_train))
        epoch_kl_loss_test.append(np.nanmean(kl_loss_test))
        epoch_ce_loss_train.append(np.nanmean(ce_loss_train))
        epoch_ce_loss_test.append(np.nanmean(ce_loss_test))

        writer.add_scalar('Epochs/loss_train', np.nanmean(loss_train), epoch)
        writer.add_scalar('Epochs/acc_train', acc_train_, epoch)
        writer.add_scalar('Epochs/loss_test', np.nanmean(loss_test), epoch)
        writer.add_scalar('Epochs/acc_val', acc_test_, epoch)

        writer.add_scalar('Epochs/kl_loss_train', np.nanmean(kl_loss_train), epoch)
        writer.add_scalar('Epochs/kl_loss_test', np.nanmean(kl_loss_test), epoch)
        writer.add_scalar('Epochs/ce_loss_train', np.nanmean(ce_loss_train), epoch)
        writer.add_scalar('Epochs/ce_loss_test', np.nanmean(ce_loss_test), epoch)

        mlflow.log_metric('loss_train', np.nanmean(loss_train))
        mlflow.log_metric('acc_train', acc_train_)
        mlflow.log_metric('loss_test', np.nanmean(loss_test))
        mlflow.log_metric('acc_val', acc_test_)

        mlflow.log_metric('kl_loss_train', np.nanmean(kl_loss_train))
        mlflow.log_metric('kl_loss_test', np.nanmean(kl_loss_test))
        mlflow.log_metric('ce_loss_train', np.nanmean(ce_loss_train))
        mlflow.log_metric('ce_loss_test', np.nanmean(ce_loss_test))


        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(epoch, np.nanmean(loss_test), model, l_plot)
        if config.model != 'FixedGraphInGraph' and config.dataset not in ['HCP', 'Tox21', 'mnist'] and not config.DGCNN:
            scheduler.step()

        else:
            if config.dataset in ['Tox21', 'HCP', 'mnist']:
                scheduler.step(np.nanmean(loss_test))
            else:
                scheduler.step()

        print("Current lr:", optimizer.param_groups[0]['lr'])


        if early_stopping.early_stop:
            torch.save((epoch_loss_train, epoch_loss_test,
                        epoch_kl_loss_train, epoch_kl_loss_test,
                        epoch_add_loss_train, epoch_add_loss_test,
                        epoch_ce_loss_train, epoch_ce_loss_test,
                        epoch_acc_train, epoch_acc_test,
                        l_plot), config.arg.saving_path + "/losses.pt")
            print("Early stopping")

            break

    # return best model
    checkpoint = torch.load(config.arg.saving_path + "/checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(config.device)
    return model, fixed_test_graphs

def eval_model(config, model, loader_test, fixed_test_graphs):
    model.eval()
    acc_test = []
    roc_auc_score_test = []
    for j, data in enumerate(loader_test):
        data = data.to(config.device)
        if config.DGCNN:
            prediction_probs, edge_index, x = model(data)
            acc, roc_auc_score_value = compute_metrics(data, prediction_probs, config=config)
        elif config.model == 'FixedGraphInGraph':
            prediction_probs = model(data, fixed_test_graphs[j])
            acc, roc_auc_score_value = compute_metrics(data, prediction_probs, config=config)
        else:
            prediction_probs, g_x, edge_index, edge_weight, adj, pytorch_target_distr = model(data)
            acc, roc_auc_score_value = compute_metrics(data, prediction_probs, adj=adj, config=config)

        acc_test.append(acc)
        roc_auc_score_test.append(roc_auc_score_value)
    if config.model != 'FixedGraphInGraph' and not config.DGCNN:
        print("pytorch_target_distr", pytorch_target_distr)
        print("adj", adj)

    return np.nanmean(acc_test), np.nanmean(roc_auc_score_test)

def train_gcn_model(config, loader_train, loader_val, loader_test):
    mlflow.log_param('dataset', config.dataset)

    model = GCN(config)
    model = model.to(config.device)
    fixed_test_graphs = []
    print("Current model", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr
                                 # , weight_decay=1e-3#, amsgrad=True  # , weight_decay=0 -4
                                 )

    if config.dataset=='HCP' :

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
                                          threshold_mode='abs')
    elif config.dataset in ['PROTEINS_full', 'mnist']:

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
                                          threshold_mode='abs')
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

    writer = SummaryWriter(config.arg.writer_path)

    criterion = define_loss(config)

    # Load trained model from path=config.arg.saving_path
    initial_epoch = 0
    continue_training = False
    if os.path.exists(config.arg.saving_path + "/checkpoint"):
        checkpoint = torch.load(config.arg.saving_path + "/checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        initial_epoch = checkpoint['epoch']
        continue_training = True


    early_stopping = EarlyStopping(patience=config.patience, verbose=True,
                                   path=config.arg.saving_path, config=config, continue_training=continue_training)


    # run training
    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_test = []
    epoch_acc_test = []

    epoch_kl_loss_train = []
    epoch_kl_loss_test = []
    epoch_add_loss_train = []
    epoch_add_loss_test = []
    epoch_ce_loss_train = []
    epoch_ce_loss_test = []

    for epoch in range(initial_epoch, config.n_epoch):
        l_plot = []

        t = time.time()

        loss_train = []
        acc_train = []
        loss_test = []
        acc_test = []




        roc_auc_score_test = []
        roc_auc_score_value_train = []

        model.train()
        for j, data in enumerate(loader_train):
            optimizer.zero_grad()

            data = data.to(config.device)


            if config.DGCNN:
                prediction_probs, _, _ = model(data)
            else:
                prediction_probs = model(data)

            train_loss, _ = compute_losses(config, data, prediction_probs, criterion)


            train_loss.backward()
            optimizer.step()

            acc, roc_auc_score_value = compute_metrics(data, prediction_probs, config=config)
            acc_train.append(acc)
            roc_auc_score_value_train.append(roc_auc_score_value)
            loss_train.append(train_loss.item())
            writer.add_scalar('Loss/train', train_loss.item())
            writer.add_scalar('Acc/train', acc)

        model.eval()

        for j, data in enumerate(loader_val):
            data = data.to(config.device)
            if config.DGCNN:
                prediction_probs, _, _ = model(data)
            else:
                prediction_probs = model(data)

            test_loss, _ = compute_losses(config, data, prediction_probs, criterion)



            acc, roc_auc_score_value = compute_metrics(data, prediction_probs,  config=config)


            acc_test.append(acc)



            roc_auc_score_test.append(roc_auc_score_value)
            loss_test.append(test_loss.item())

            writer.add_scalar('Loss/val', test_loss.item())
            writer.add_scalar('Acc/val', acc)




        acc_train_ = np.nanmean(np.array(acc_train))
        acc_test_ = np.nanmean(np.array(acc_test))

        print('%d] loss train: %.2e loss test: %.2e acc: %.2f  acc_val: %.2f  (%.2f s)' % (
            epoch, np.nanmean(loss_train), np.nanmean(loss_test), acc_train_, acc_test_, (time.time() - t)))

        print('%d] roc_auc_score_train: %.2f  roc_auc_score_test: %.2f (%.2f s)' % (
                  epoch,
                  np.nanmean(roc_auc_score_value_train), np.nanmean(roc_auc_score_test), (time.time() - t)))

        epoch_loss_train.append(np.nanmean(loss_train))
        epoch_acc_train.append(acc_train_)
        epoch_loss_test.append(np.nanmean(loss_test))
        epoch_acc_test.append(acc_test_)



        writer.add_scalar('Epochs/loss_train', np.nanmean(loss_train), epoch)
        writer.add_scalar('Epochs/acc_train', acc_train_, epoch)
        writer.add_scalar('Epochs/loss_test', np.nanmean(loss_test), epoch)
        writer.add_scalar('Epochs/acc_test', acc_test_, epoch)


        mlflow.log_metric('loss_train', np.nanmean(loss_train))
        mlflow.log_metric('acc_train', acc_train_)
        mlflow.log_metric('loss_test', np.nanmean(loss_test))
        mlflow.log_metric('acc_test', acc_test_)




        early_stopping(epoch, np.nanmean(loss_test), model, l_plot)

        if config.dataset=='HCP':
            scheduler.step(np.nanmean(loss_test))
        elif config.dataset in ['PROTEINS_full', 'mnist']:
            scheduler.step(np.nanmean(loss_test))
        else:
            scheduler.step()

        print("Current lr:", optimizer.param_groups[0]['lr'])

        if early_stopping.early_stop:
            torch.save((epoch_loss_train, epoch_loss_test,
                        epoch_kl_loss_train, epoch_kl_loss_test,
                        epoch_add_loss_train, epoch_add_loss_test,
                        epoch_ce_loss_train, epoch_ce_loss_test,
                        epoch_acc_train, epoch_acc_test,
                        l_plot), config.arg.saving_path + "/losses.pt")
            print("Early stopping")

            break

    # return best model
    checkpoint = torch.load(config.arg.saving_path + "/checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(config.device)
    return model, fixed_test_graphs

def eval_gcn_model(config, model, loader_test, fixed_test_graphs):
    model.eval()
    acc_test = []
    roc_auc_score_test = []
    for j, data in enumerate(loader_test):
        data = data.to(config.device)
        if config.DGCNN:
            prediction_probs, _, _ = model(data)
        else:
            prediction_probs = model(data)

        acc, roc_auc_score_value = compute_metrics(data, prediction_probs, config=config)

        acc_test.append(acc)
        roc_auc_score_test.append(roc_auc_score_value)


    return np.nanmean(acc_test), np.nanmean(roc_auc_score_test)


import pytorch_lightning as pl
import torchmetrics
from models import GiG
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from common_utils import SoftHistogram, compute_clustering_coeff,\
    MultiTaskBCELoss, mask_nans
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
import os.path
import shutil
import sklearn
from contextlib import suppress

losses = nn.ModuleDict({
                'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
                'CrossEntropyLoss': nn.CrossEntropyLoss(),
                'MultiTaskBCE': MultiTaskBCELoss()
        })

class LglKlTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters(self.config)
        self.model = GiG(self.config)
        self.distribution_type = config["target_distribution"]
        self.initial_loss = losses[config["loss"]]
        self.saving_path = config["saving_path"]+'plots/'
        if os.path.exists(self.saving_path):
            shutil.rmtree(self.saving_path)
        access = 0o777
        os.makedirs(self.saving_path, access)

        self.alpha = config["alpha"]

    def training_step(self, batch, batch_idx, optimizer_idx):

        metrics = self._shared_step(batch, addition="train")
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, addition="val")
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, addition="test")
        self.log_dict(metrics)
        # if not os.path.isfile(self.saving_path + 'G_trsh_' + str(0.5) + '.graphml'):
        #     if batch_idx == 0:
        #         for trh in [0, 0.5, 0.3, 0.1, 0.01, 0.001, 0.0001]:
        #             self._save_plottings(batch, metrics, self.saving_path, trh)
        return metrics

    def _shared_step(self, data, addition):
        y_hat, feature_matrix, edge_index, edge_weight, adj_matrix = self.model(data)
        # adj_matrix = torch.squeeze(to_dense_adj(edge_index, edge_attr=edge_weight))
        if self.config["loss"] == "CrossEntropyLoss":
            loss = self.initial_loss(y_hat, data.y.long())
            y_hat = torch.softmax(y_hat,dim=1)
            labels = data.y

        elif self.config["dataset"] == 'Tox21':
            loss = self.initial_loss(y_hat, data.y)
            y_hat = torch.sigmoid(y_hat)


        else:
            loss = self.initial_loss(y_hat, data.y.float())
            y_hat = torch.sigmoid(y_hat)
            labels = data.y


        if self.config["dataset"]=='Tox21':
            y_hat, labels = mask_nans(y_hat,data.y)
        labels = labels.int()

        # try:
        n_nodes = adj_matrix.shape[0]
        softhist = SoftHistogram(bins=n_nodes, min=0.5, max=n_nodes + 0.5, sigma=0.6)
        kl_loss = self.alpha * self._compute_kl_loss(adj_matrix, n_nodes, softhist)
        loss += kl_loss
        avg_coeff, eigenDecomposition_coeff = compute_clustering_coeff(np.array(adj_matrix.clone().detach().cpu()))


        acc = torchmetrics.functional.accuracy(y_hat, labels)
        f1 = torchmetrics.functional.f1(y_hat, labels)
        precision = torchmetrics.functional.precision(y_hat, labels)
        recall = torchmetrics.functional.recall(y_hat, labels)
        metrics = {addition + "_acc": acc, addition + "_f1": f1,
                   # addition + "_auc": auc,
                   # addition + "_auroc": auroc,
                   addition + "_precision": precision,
                   addition + "_recall": recall, addition + "_avg_coeff": avg_coeff,
                   addition + "_eigenDecomposition_coeff": eigenDecomposition_coeff,
                   "loss": loss,
            # , "kl_loss": kl_loss.clone().detach(),
                   addition + "_loss": loss
                   }
        if addition == "test":
            # TODO change to compute it at the end of epoch at_epoch_end
            # https: // github.com / PyTorchLightning / pytorch - lightning / issues / 2210
            # auc = torchmetrics.functional.auc(y_hat, data.y, reorder=True)
            # auroc = torchmetrics.functional.auroc(y_hat, data.y)
            # auc = sklearn.metrics.auc(np.array(data.y),np.array(y_hat))
            # auroc = sklearn.metrics.roc_auc_score(np.array(data.y.detach().cpu()),np.array(y_hat.detach().cpu()))
            metrics["adj"] = adj_matrix
            metrics["y_hat"] = torch.sigmoid(y_hat)
            # metrics[addition + "_auc"] = auc
            # metrics[addition + "_auroc"] = auroc
        return metrics

    def _kl_div(self, p, q):
        return torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8))

    def _compute_distr(self, adj, softhist):
        deg = adj.sum(-1)
        distr = softhist(deg)
        return distr / torch.sum(distr), deg

    def _compute_kl_loss(self, adj, batch_size, softhist):
        binarized_adj = torch.zeros(adj.shape).to(adj.device)
        binarized_adj[adj > 0.5] = 1
        dist, deg = self._compute_distr(adj * binarized_adj, softhist)
        target_dist = self._compute_target_distribution(batch_size)
        kl_loss = self._kl_div(dist, target_dist)
        return kl_loss

    def _compute_target_distribution(self, batch_size):
        # define target distribution
        target_distribution = torch.zeros(batch_size).to(self.model.population_level_module.sigma.device)
        # TODO: check if it is valid idea
        if batch_size > 4:
            tab = 4
        else:
            tab = 0
        if self.distribution_type == 'power_law':

            target_distribution[tab:] = self.model.population_level_module.sigma * (
                    1.0 + torch.arange(batch_size - tab).to(self.model.population_level_module.sigma.device)).pow(self.model.population_level_module.mu)
        else:
            target_distribution[tab:] = torch.exp(
                -(self.model.population_level_module.mu - torch.arange(batch_size - tab).to(self.model.population_level_module.sigma.device) + 1.0) ** 2 / (
                        self.model.population_level_module.sigma ** 2))

        return target_distribution / target_distribution.sum()

    # def predict_step(self, batch, batch_idx, dataloader_idx, saving_path):
    #     metrics = self._shared_step(batch, addition="test")
    #     self._save_plottings(batch, metrics, saving_path, trh)
    #     # self.log_dict(metrics)

    def configure_optimizers(self):
        if self.config["optimizer_type"] == 'adam':
            # optimizer
            population_level_module_par = [param for name_, param in
                                           self.model.population_level_module.named_parameters()
                                           if name_ not in ['temp', 'theta', 'mu', 'sigma']]
            population_level_module_par.extend(self.model.node_level_module.parameters())
            population_level_module_par.extend(self.model.gnn.parameters())
            population_level_module_par.extend(self.model.classifier.parameters())

            optimizer = torch.optim.Adam(population_level_module_par, lr=self.config["optimizer_lr"])
            kl_loss_optimizer = torch.optim.Adam([self.model.population_level_module.mu,
                                                  self.model.population_level_module.sigma,
                                                  ], lr=self.config["lr_mu_sigma"]
                                                 )
            lgl_optimizer = torch.optim.Adam([self.model.population_level_module.theta,
                                              self.model.population_level_module.temp,
                                              ], lr=self.config["lr_theta_temp"]
                                             )
        else:
            print("Not implemented")

        # if self.config["dataset"] in ['Tox21', 'HCP', 'MUTAG', 'REDDIT-BINARY']:
        if self.config["scheduler"] == 'ReduceLROnPlateau':

            scheduler = {"scheduler": ReduceLROnPlateau(
                    optimizer, patience=10,
                    threshold=0.0001,
                    mode='min', verbose=True, threshold_mode='abs'),
                "interval": "epoch",
                # "monitor": "val_loss"}
                "monitor": "loss"}

            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
            #                               threshold_mode='abs')
        elif self.config["scheduler"] == 'CosineAnnealingLR':
            scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=10),
                "interval": "epoch",
                # "monitor": "val_loss"}
                "monitor": "loss"}

            # scheduler = CosineAnnealingLR(optimizer, T_max=10)
        else:
            print("This scheduler is not implemented.")

        return [optimizer, kl_loss_optimizer, lgl_optimizer], scheduler

    def _save_plottings(self, data, metrics, saving_path, trh):
        adj = metrics["adj"]
        soft_pred = metrics["y_hat"]

        n_nodes = adj.shape[0]
        target_distr = self._compute_target_distribution(n_nodes)

        deg = adj.sum(-1)
        deg_bin = (adj > trh).sum(-1)  # binarized adj matrix

        hist = torch.histc(deg, bins=n_nodes, min=0.5, max=n_nodes + 0.5)
        hist = hist / hist.sum()

        hist_bin = torch.histc(deg_bin.float(), bins=n_nodes, min=0.5, max=n_nodes + 0.5)
        hist_bin = hist_bin / hist_bin.sum()

        # sns_plot = sns.histplot(np.array((adj > trh).detach().cpu().view(-1)), bins=10, stat="probability").figure
        # sns_plot.savefig(saving_path + 'weight_distr_' + str(trh) + '.png', bbox_inches="tight")
        # adj_array =np.array((adj > trh).detach().cpu().view(-1), dtype='f')
        # # weights = np.ones_like(adj_array) / (len(adj_array)*10)
        # plt.hist(adj_array, bins=10, density=True, )
        # plt.savefig(saving_path + 'weight_distr_' + str(trh) + '.png', bbox_inches="tight")

        sns.histplot(np.array((adj > trh).detach().cpu().view(-1), dtype='f'), bins=10, stat="probability")
        plt.savefig(saving_path + 'weight_distr_' + str(trh) + '.png', bbox_inches="tight")

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(22, 4))
        ax1.set_title('Learned target distribution')
        ax2.set_title('Sum of rows of adjacency matrix')
        ax3.set_title('Node degrees of binarized adjacency matrix')
        ax1.bar(np.arange(n_nodes), (target_distr[:n_nodes].detach().cpu() * 100).int())
        ax2.bar(np.arange(len(hist)), (hist.cpu() * 100).int().numpy())
        ax3.bar(np.arange(len(hist_bin)), (hist_bin.cpu() * 100).int().numpy())
        plt.savefig(saving_path + 'G_trsh' + str(trh) + '.png', bbox_inches="tight")
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
        ax1.set_title('adj')
        ax2.set_title('binarized adj')
        ax1.imshow(adj[:n_nodes, :n_nodes].detach().cpu())
        ax2.imshow((adj[:n_nodes, :n_nodes] > trh).detach().cpu())
        plt.savefig(saving_path + 'G_binarized_' + str(trh) + '.png', bbox_inches="tight")
        plt.clf()

        adj_ = torch.zeros(adj.shape)
        adj_[adj > trh] = 1
        # create_graph_plots(self.config, adj_, data, soft_pred, trh, saving_path)
class LglTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        self.model = GiG(self.config)
        self.initial_loss = losses[config["loss"]]
        self.saving_path = config["saving_path"]

    def training_step(self, batch, batch_idx, optimizer_idx):
        metrics, _  = self._shared_step(batch, addition="train")
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics, _ = self._shared_step(batch, addition="val")
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics, adj = self._shared_step(batch, addition="test")
        self.log_dict(metrics)
        if not os.path.isfile(self.saving_path + 'G_trsh_' + str(0.5) + '.graphml'):
            if batch_idx == 0:
                for trh in [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001]:
                    self._save_plottings(batch, metrics, adj,  self.saving_path, trh)
        return metrics

    def _shared_step(self, data, addition):
        y_hat, feature_matrix, edge_index, edge_weight, adj_matrix = self.model(data)
        adj_matrix = torch.squeeze(to_dense_adj(edge_index, edge_attr=edge_weight))
        try:
            adj_matrix = np.array(adj_matrix.detach().cpu())
            avg_coeff, eigenDecomposition_coeff = compute_clustering_coeff(adj_matrix)
        except:
            avg_coeff = 0.0
            eigenDecomposition_coeff = 0.0
        if self.config["loss"] == "CrossEntropyLoss":
            loss = self.initial_loss(y_hat, data.y.long())
            y_hat = torch.softmax(y_hat,dim=1)
            labels = data.y


        elif self.config["dataset"] == 'Tox21':
            loss = self.initial_loss(y_hat, data.y)
            y_hat = torch.sigmoid(y_hat)


        else:
            loss = self.initial_loss(y_hat, data.y.float())
            y_hat = torch.sigmoid(y_hat)
            labels = data.y


        if self.config["dataset"]=='Tox21':
            y_hat, labels = mask_nans(y_hat,data.y)
        labels = labels.int()



        acc = torchmetrics.functional.accuracy(y_hat, labels)
        f1 = torchmetrics.functional.f1(y_hat, labels)
        # auc = torchmetrics.functional.auc(y_hat, data.y, reorder=True)
        # auroc = torchmetrics.functional.auroc(y_hat, data.y)
        precision = torchmetrics.functional.precision(y_hat, labels)
        recall = torchmetrics.functional.recall(y_hat, labels)
        metrics = {addition+"_acc": acc, addition+"_f1": f1,
                   # addition+"_auc": auc,
                   # addition+"_auroc": auroc,
                   addition+"_precision": precision,
                   addition+"_recall": recall, addition+"_avg_coeff": avg_coeff,
                   addition+"_eigenDecomposition_coeff": eigenDecomposition_coeff,
                   addition+"_loss": loss.clone().detach(), "loss": loss}
        if addition == "test":
            # metrics["adj"] = adj_matrix
            metrics["y_hat"] = torch.sigmoid(y_hat)
        return metrics, adj_matrix

    # def predict_step(self, batch, batch_idx, dataloader_idx):
    #     metrics = self._shared_step(batch, addition="predict")
    #     return metrics

    def configure_optimizers(self):
        optimizers =[]
        if self.config["optimizer_type"] == 'adam':
            # optimizer
            population_level_module_par = [param for name_, param in
                                           self.model.population_level_module.named_parameters()
                                           if name_ not in ['temp', 'theta', 'mu', 'sigma']]
            population_level_module_par.extend(self.model.node_level_module.parameters())
            population_level_module_par.extend(self.model.gnn.parameters())
            population_level_module_par.extend(self.model.classifier.parameters())

            optimizer = torch.optim.Adam(population_level_module_par, lr=self.config["optimizer_lr"])
            optimizers.append(optimizer)
            if self.config["population_level_module"] != "DGCNN":
                lgl_optimizer = torch.optim.Adam([self.model.population_level_module.theta,
                                                  self.model.population_level_module.temp,
                                                  ], lr=self.config["lr_theta_temp"]
                                                 )
                optimizers.append(lgl_optimizer)

        else:
            print("Not implemented")

        if self.config["dataset"] in ['Tox21', 'HCP']:
            scheduler = {"scheduler": ReduceLROnPlateau(
                optimizer, patience=10,
                threshold=0.0001,
                mode='min', verbose=True, threshold_mode='abs'),
                "interval": "epoch",
                "monitor": "val_loss"}

            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
            #                               threshold_mode='abs')
        else:
            scheduler = {"scheduler": CosineAnnealingLR(optimizer, T_max=10),
                         "interval": "epoch",
                         "monitor": "val_loss"}

        return optimizers, [scheduler]

    def _save_plottings(self, data, metrics, adj, saving_path, trh):
        soft_pred = metrics["y_hat"]
        adj = torch.squeeze(torch.tensor(adj))
        adj_ = torch.zeros(adj.shape)
        adj_[adj > trh] = 1
        # create_graph_plots(self.config, adj_, data, soft_pred, trh, saving_path)


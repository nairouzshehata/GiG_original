#
# class MLConfig:
#     def __init__(self, arg):
#         self.arg = arg
#         dataset = self.arg.dataset
#
#
#         print(dataset)
#
#         if dataset == "PROTEINS_full":
#             self.dataset = dataset
#             self.batch_size_train = 50
#             self.batch_size_val = 64
#             self.batch_size_test = 50
#             self.lr = 1e-3# wo kl
#
#             self.sigma_lr =1e-2 # should be the same as lr
#             self.mu_lr =1e-2
#             self.theta_lr = 1e-2
#             self.temp_lr = 1e-2
#             self.temp_initial =1e-4
#             self.theta_initial = 8.0#8.0
#
#             self.pooling = 'mean'  # , add 'mean', 'attention', 'set2set'
#             self.random_split = False
#
#             # percentage of training set to use as validation
#             self.n_epoch = 3000
#             self.input_dim = (28 * 28)
#             self.output_dim = 10
#             self.random_seed = 0
#             self.num_node_features = 1
#             self.device = 'cuda'#'cpu'
#             self.dropout = 0.15  # 0.15 # try 0 drop out
#             self.use_dropout = False
#             self.fix_population_level = False
#             self.fix_node_level = True
#             self.fix_dynamic_node_convolution = False
#             self.writer_path = "runs/"
#             self.kl_patience = 12
#             self.ce_patience = 5
#             self.patience = 50
#
#
#             self.feature_dataset = False
#             self.data_level = 'graph'  # 'image', 'feature_matrix','graph'
#             self.k = '3'  # which folder to use
#
#
#
#             self.DGCNN = False  # uses PyCDGM with nodeconv
#             self.regularization = ''  #''#'KL', 'homogenious', 'None', 'L1', 'KL_adj_dist', 'KL_degree_dist'
#             if self.regularization == 'KL_degree_dist':
#                 self.theta_initial = 1.0  # 8.0
#             self.model = 'FixedGraphInGraph'  # MLP, FixedGraphInGraph, GCN
#             self.fix_population_graph = 'knn' # 'knn', 'random'
#             self.target_distribution = 'normal' #self.arg.target_distribution#'normal'# 'normal', 'power_law'
#             if self.target_distribution == 'normal':
#                 self.kl_sum_par = 0.003
#                 self.kl_sum_par_after_tresh = 0.003# 0.05 most working version 0.88 nwas with 0.02
#
#             elif self.target_distribution == 'power_law':
#                 self.kl_sum_par = 0.3
#
#             else:
#                 self.kl_sum_par = ''
#             self.ce_sum_par = 1
#             self.ce_sum_par_after_tresh = 1
#             self.learnable_distr = True
#             self.kl_treshold = 2
#             self.fix_graph_random_num = 0 # should start from 0!
#             self.sigma_reg = False
#

import yaml
import os
from dataclasses import dataclass
import torch







# class Config:
#     def __init__(self, yaml_config):
#
#
#
#         self.dataset = yaml_config["dataset_name"]
#         # Define models layers
#         model_layers = yaml_config["model"]
#         self.population_level_module = model_layers["population_level_module"]["type"]
#         self.node_level_module = model_layers["node_level_module"]["type"]
#         self.population_layers = model_layers["population_level_module"][self.population_level_module][
#             "layers_sizes"]  # layers
#         self.node_layers = model_layers["node_level_module"][self.node_level_module]["layers_sizes"]
#         self.gnn_layers = model_layers["GNN"]["layers_sizes"]
#         self.classifier_layers = model_layers["classifier"]["layers_sizes"]
#         # Define models parameters
#         self.model_params = {"pooling": yaml_config["model_parameters"]["pooling"]}
#         # Define training parameters
#         self.training_params = {"epochs": yaml_config["training_parameters"]["epochs"],
#                                 "patience": yaml_config["training_parameters"]["patience"]
#                                 }
#         # Data information
#         self.data_parameters = {"output_dim": yaml_config["data_parameters"]["output_dim"],
#                                 "input_dim": yaml_config["data_parameters"]["input_dim"],
#                                 "num_node_features": yaml_config["data_parameters"]["num_node_features"],
#                                 "bs_train": yaml_config["data_parameters"]["batch_size"]["train"],
#                                 "bs_val": yaml_config["data_parameters"]["batch_size"]["val"],
#                                 "bs_test": yaml_config["data_parameters"]["batch_size"]["test"],
#                                 "split_type": yaml_config["data_parameters"]["splits"]["type"],
#                                 "train_split": yaml_config["data_parameters"]["splits"]["train"],
#                                 "val_split": yaml_config["data_parameters"]["splits"]["val"],
#                                 "test_split": yaml_config["data_parameters"]["splits"]["test"]
#                                 }
#         # Loss and optimizer parameters
#         self.optimizer = {"type": yaml_config["training_parameters"]["optimizer"]["type"],
#                           "lr": float(yaml_config["training_parameters"]["optimizer"]["lr"])}
#         if self.population_level_module == "LGL":
#             self.optimizer = {**self.optimizer, **{"lr_theta_temp": float(yaml_config["LGL"]["lr_theta_temp"])}}
#             self.model_params = {**self.model_params, **{"temp": float(yaml_config["LGL"]["temp"]),
#                                                          "theta": float(yaml_config["LGL"]["theta"])}}
#         if self.population_level_module == "LGLKL":
#             self.optimizer = {**self.optimizer, **{"lr_theta_temp": float(yaml_config["LGLKL"]["lr_theta_temp"]),
#                                                    "lr_mu_sigma": float(yaml_config["LGLKL"]["lr_mu_sigma"]),
#                                                    "kl_params_patience": yaml_config["LGLKL"]["kl_patience"]}}
#
#             target_distribution = yaml_config["LGLKL"]["target_distribution"]
#             self.loss_params = {"target_distribution": target_distribution,
#                                 "mu": float(yaml_config["LGLKL"][target_distribution]["mu"]),
#                                 "sigma": float(yaml_config["LGLKL"][target_distribution]["sigma"]),
#                                 "alpha": float(yaml_config["LGLKL"][target_distribution]["alpha"]),
#                                 "loss": yaml_config["training_parameters"]["loss"]["type"],
#                                 "loss_weights": yaml_config["training_parameters"]["loss"]["weights"]
#                                 }
#
#             self.model_params = {**self.model_params, **{"temp": float(yaml_config["LGLKL"]["temp"]),
#                                                          "theta": float(yaml_config["LGLKL"]["theta"])}}
#         if self.population_level_module == "DGCNN":
#             self.model_params = {**self.model_params, **{"k": yaml_config["DGCNN"]["k"]}}
#
#         # Saving paths
#         self.dataset_path = yaml_config["paths"]["dataset_path"]
#         self.saving_path = yaml_config["paths"]["model_path"] + self.dataset + "/" + \
#                            self.node_level_module + "_" + self.population_level_module + "/"


def create_config(config_name, model_type=None, population_level_module_type=None,
                  alpha=None, pooling=None, gnn_type=None):
    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join("configs/", config_name)) as file:
            config = yaml.safe_load(file)

        return config

    yaml_config = load_config(config_name)
    model =yaml_config["model"]
    node_level_module = model["node_level_module"]["type"]
    if not model_type:
        model_type = yaml_config["model"]["type"]
    if not population_level_module_type:
        population_level_module_type = model["population_level_module"]["type"]
    if model_type == 'GCN':
        population_level_module_type = 'none'
    if not pooling:
        pooling = yaml_config["model_parameters"]["pooling"]
    if not gnn_type:
        gnn_type = model["GNN"]["type"]

    d = {"model_type": model_type,
        "dataset": yaml_config["dataset_name"],
        "population_level_module": population_level_module_type,
        "node_level_module" : node_level_module,
        "gnn_type": gnn_type,
        "node_layers" : model["node_level_module"][node_level_module]["layers_sizes"],
        "gnn_layers" : model["GNN"][gnn_type]["layers_sizes"],
        "classifier_layers" : model["classifier"]["layers_sizes"],
        "pooling": pooling,
        "epochs": yaml_config["training_parameters"]["epochs"],
        "patience": yaml_config["training_parameters"]["patience"],
        "output_dim": yaml_config["data_parameters"]["output_dim"],
        "input_dim": yaml_config["data_parameters"]["input_dim"],
        "num_node_features": yaml_config["data_parameters"]["num_node_features"],
        "bs_train": yaml_config["data_parameters"]["batch_size"]["train"],
        "bs_val": yaml_config["data_parameters"]["batch_size"]["val"],
        "bs_test": yaml_config["data_parameters"]["batch_size"]["test"],
        "split_type": yaml_config["data_parameters"]["splits"]["type"],
        "train_split": yaml_config["data_parameters"]["splits"]["train"],
        "val_split": yaml_config["data_parameters"]["splits"]["val"],
        "test_split": yaml_config["data_parameters"]["splits"]["test"],
        "optimizer_type": yaml_config["training_parameters"]["optimizer"]["type"],
        "optimizer_lr": float(yaml_config["training_parameters"]["optimizer"]["lr"]),
        "loss": yaml_config["training_parameters"]["loss"]["type"],
        "loss_weights": yaml_config["training_parameters"]["loss"]["weights"],
        "scheduler": yaml_config["training_parameters"]["scheduler"]
    }

    if model_type != 'GCN':
        if population_level_module_type not in ["random", "knn"]:
             d = {**d, **{"population_layers": model["population_level_module"][population_level_module_type][
            "layers_sizes"]}}
        else:
            d = {**d, **{"k": model["population_level_module"][population_level_module_type]["k"]}}

    if gnn_type != 'GCN_kipf':
        d = {**d, **{"gnn_aggr": model["GNN"][gnn_type]["agg"]}}

    if population_level_module_type == "LGL":
        d = {**d, **{"lr_theta_temp": float(yaml_config["LGL"]["lr_theta_temp"])},
             **{"temp": float(yaml_config["LGL"]["temp"]),
                                                 "theta": float(yaml_config["LGL"]["theta"])}}
    if population_level_module_type == "LGLKL":
        d = {**d, **{"lr_theta_temp": float(yaml_config["LGLKL"]["lr_theta_temp"]),
                                               "lr_mu_sigma": float(yaml_config["LGLKL"]["lr_mu_sigma"])
                                               }}

        target_distribution = yaml_config["LGLKL"]["target_distribution"]
        if not alpha:
            alpha = float(yaml_config["LGLKL"][target_distribution]["alpha"])
        d = {**d, **{"target_distribution": target_distribution,
                            "mu": float(yaml_config["LGLKL"][target_distribution]["mu"]),
                            "sigma": float(yaml_config["LGLKL"][target_distribution]["sigma"]),
                            "alpha": alpha,
                             "temp": float(yaml_config["LGLKL"]["temp"]),
                             "theta": float(yaml_config["LGLKL"]["theta"])
                            }}
    if node_level_module == 'GIN':
        d["node_level_hidden_layers_number"] = model["node_level_module"][node_level_module]["hidden_layers_num"]


    if population_level_module_type == "DGCNN":
        d = {**d, **{"k": model["population_level_module"]["DGCNN"]["k"]}}

    # Saving paths
    d["dataset_path"] = yaml_config["paths"]["dataset_path"]
    d["saving_path"] = yaml_config["paths"]["model_path"] + '/'+ d["dataset"] + \
                       "/" + model_type +'_'+ population_level_module_type + "/"
    return d




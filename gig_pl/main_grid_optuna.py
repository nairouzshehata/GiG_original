from config import create_config
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import json
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
import torch
from optuna.trial import TrialState
from model_utils import *
import argparse
import os
PERCENT_VALID_EXAMPLES = 1 # % of the total validation data, limiting no. of validation batches used during validation. 
EPOCHS = 27
N_TRIALS = 60
N_RUNS = 3
# max epochs is located in config

'''ChatGPT n_trials: 
This refers to the number of trials or experiments that Optuna will run to 
find the best set of hyperparameters. Each trial represents a combination of 
hyperparameters that are evaluated.

study epochs: In this context, "study epochs" could be interpreted as the 
iterations or epochs of the optimization study conducted by Optuna. 
Each trial involves training and evaluating the model for a certain number of epochs. 
This is defined by the max_epochs parameter in the Trainer object.

runs: In the code, "runs" seems to refer to the number of times the entire 
optimization process (including multiple trials) is repeated. This allows for
robustness and validation of the hyperparameters found across multiple iterations.'''


class Objective(object):
    def __init__(self, task_name, dataset, fold_id, config):
        # Hold this implementation specific arguments as the fields of the class.
        self.dataset = dataset
        self.fold_id = fold_id
        self.config = config
        self.task_name = task_name


    def __call__(self, trial):
        drop_last_val = False
        # Calculate an objective value by using the extra arguments.
        self.config["population_level_module"]=self.task_name
        self.config["bs_train"] = trial.suggest_int("bs_train", 25, 100)
        self.config["node_level_hidden_layers_number"] = trial.suggest_int("node_level_hidden_layers_number", 2, 7)
        if self.config["node_level_module"] == "GIN":
            self.config["n_node_layers"] = 2
        else:
            self.config["n_node_layers"] = trial.suggest_int("n_node_layers", 2, 5)
        self.config["theta"] = trial.suggest_float("theta", 0.5, 11.0)
        if self.task_name == 'LGLKL':
            self.config["alpha"] = trial.suggest_float("alpha", 0.001,  0.3)
        self.config["pooling"] = trial.suggest_categorical("pooling",['mean', 'add'])
        self.config["optimizer_lr"] = trial.suggest_float("optimizer_lr", 1e-4, 1e-1, log=True)
        self.config["node_layers"] =[
            trial.suggest_int("node_layers_n_units_l{}".format(i), config["num_node_features"], 300, log=True) for i in range( self.config["n_node_layers"])
        ]
        gnn_layers_node_layers = trial.suggest_int("gnn_layers_node_layers", 2, 5)
        self.config["gnn_layers_node_layers"]= gnn_layers_node_layers

        self.config["gnn_layers"] = [
            trial.suggest_int("gnn_layers_n_units_l{}".format(i), 20, 120, log=True) for i in range(gnn_layers_node_layers)
        ]



        self.train_loader, self.val_loader = self.dataset.get_model_selection_fold(outer_idx=self.fold_id,
                                                                    batch_size_train=self.config["bs_train"],
                                                                    batch_size_val=self.config["bs_val"],
                                                                    shuffle=True, drop_last_val=drop_last_val)

        trainer = pl.Trainer(
            logger=True,
            limit_val_batches=PERCENT_VALID_EXAMPLES,
            enable_checkpointing=True,
            max_epochs=EPOCHS,
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        )

        task = define_task(self.task_name, self.config)

        trainer.fit(task, self.train_loader, self.val_loader)
        if 'val_acc' in trainer.callback_metrics:
            final_result = trainer.callback_metrics["val_acc"].item()
        return final_result


def saved_hyperparameters_exist():
    """
    Check if saved hyperparameters exist.
    
    Returns:
        bool: True if saved hyperparameters exist, False otherwise.
    """
    return os.path.exists("saved_hyperparameters_json")

def load_saved_hyperparameters():
    """
    Load saved hyperparameters from file.
    
    Returns:
        dict: Dictionary containing the loaded hyperparameters.
    """
    with open("saved_hyperparameters.json", "r") as f:
        saved_hyperparameters = json.load(f)
    return saved_hyperparameters

def save_hyperparameters(config):
    """
    Save hyperparameters to a JSON file.

    Args:
        config (dict): Dictionary containing hyperparameters.
    """
    with open("saved_hyperparameters.json", "w") as f:
        json.dump(config, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' '
                                                 '')
    parser.add_argument('--dataset', default='PROTEINS_full', type=str,
                        help='Name of dataset {"DD", "NCI1",  "ENZYMES", "PROTEINS_full"}.')
    parser.add_argument('--node_level_module', default='GIN', type=str,
                        help='Name of node level module {"GraphConv", "GIN"}.')
    parser.add_argument('--population_level_module_type', default="LGL", type=str,
                        help=': '
                             '{"LGL", "LGLKL"}.')
    parser_args = parser.parse_args()


    #define the models
    dataset_name= parser_args.dataset
    # Use the same splits, so first run Prepare dataset and change the splits there!!!!!
    data_dir = 'data/CHEMICAL'
    model_type = 'GraphInGraph'
    population_level_module_type = parser_args.population_level_module_type
    results_dict = {}
    dataset = define_dataset(dataset_name=dataset_name, data_dir=data_dir)
    # Overwrites the default values
    config = create_config(config_name=DATASETS2YAML[dataset_name], model_type=model_type,
                               population_level_module_type=population_level_module_type)
    print("config", config)
    folds_ids = range(0,N_RUNS)
    # retrain on a whole dataset and test
    
    for i, fold_id in enumerate(folds_ids):
        if saved_hyperparameters_exist():
            config = load_saved_hyperparameters()
        else:

            # Execute an optimization by using an `Objective` instance.
            study = optuna.create_study(direction="maximize")
            study.optimize(Objective(task_name=population_level_module_type, dataset=dataset,
                                    fold_id=fold_id, config=config), n_trials=N_TRIALS)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                config[key] = value
                print("    {}: {}".format(key, value))

            config["node_layers"] = [trial.params["node_layers_n_units_l{}".format(i)] for i in range(config["n_node_layers"])]

            config["gnn_layers"] = [trial.params["gnn_layers_n_units_l{}".format(i)] for i in range(config["gnn_layers_node_layers"])]

            print("config", config)

            # Save the best hyperparameters
            save_hyperparameters(config)

            # retrain on a whole dataset and test
            train_loader, val_loader = dataset.get_model_selection_fold(outer_idx=fold_id,
                                                                        batch_size_train=config["bs_train"],
                                                                        batch_size_val=config["bs_val"],
                                                                        shuffle=True, drop_last_val=False)
        
            test_folder = dataset.get_test_fold(outer_idx=fold_id, batch_size=config["bs_test"], shuffle=True)

            checkpoint_callback = ModelCheckpoint(monitor="val_loss", verbose=False,
                                                save_last=False, save_top_k=1, save_weights_only=False,
                                                mode='min', every_n_epochs=1)
            early_stopping_callback = EarlyStopping(monitor="val_loss",
                                                    patience=config["patience"])

            trainer = pl.Trainer(devices=1 if torch.cuda.is_available() else None,
                                log_every_n_steps=5,
                                max_epochs=config["epochs"],
                                enable_checkpointing= True,
                                callbacks=[checkpoint_callback, early_stopping_callback],
                                deterministic=False, default_root_dir=config["saving_path"])
            print("config", config["saving_path"])
            task = define_task(task_name=population_level_module_type, config =config)
            # task = LglKlTask(config)

            trainer.fit(task, train_loader, val_loader)
            results = trainer.test(dataloaders=test_folder, ckpt_path="best")
            results_dict[fold_id] = results
            with open(config["saving_path"] + '/lightning_logs/version_'+str(i) +
                    '/results'+str(fold_id)+'.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    df = pd.DataFrame.from_dict(results_dict)
    df['mean'] = df.mean(axis=1).values
    df['std'] = df.std(axis=1).values
    df.to_csv(config["saving_path"] + 'results.csv')


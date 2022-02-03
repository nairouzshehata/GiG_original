# Graph-in-Graph (GiG)
Learning interpretable latent graphs in the non-Euclidean domain for molecule prediction and healthcare applications.

## Installation

Requirements file contains necessary packages. Please first install torch and connected packages following the instructions from [pytorch](https://pytorch.org/get-started/locally/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

We used:

```bash
torch==1.7.1
torch_cluster==1.5.8
torch_geometric==2.0.2
torchvision==0.8.2
```
However, other versions can be used as well. 

Then the rest packages can be installed by 
```bash
pip install -r requirements.txt
```

## Instructions
### Datasets
For bench-marking datasets please follow instructions from [A Fair Comparison of Graph Neural Networks for Graph Classification](
https://github.com/diningphil/gnn-comparison#instructions) or use already preprocessed version located in **data/CHEMICAL**.
Code for preprocessing datasets was taken from paper [A Fair Comparison of Graph Neural Networks for Graph Classification](
https://github.com/diningphil/gnn-comparison#instructions) [1]

### Execution 
To run the experiments with LGL model
```bash
python main_grid_optuna.py --dataset <dataset_name> --population_level_module_type <model>
```
Where ```<dataset_name>``` is one of the datasets ```DD, ENZYMES,"NCI1", "PROTEINS_full" ```
and  ```<model>``` in ``` LGL, LGLKL``` 

[1] Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli: A Fair Comparison of Graph Neural Networks for Graph Classification. Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)

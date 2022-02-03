# Graph-in-Graph (GiG)
Learning interpretable latent graphs in non-Euclidean domain for molecule prediction and healthcare applications.

## Installation

Requirements file contains necessary packages. Please fist install torch and connected packages following the instructions from [pytorch](https://pytorch.org/get-started/locally/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

We used:

```bash
torch==1.7.1
torch_cluster==1.5.8
torch_geometric==2.0.2
torchvision==0.8.2
```
However, another versions can be used as well. 

Then the rest packages can be installed by 
```bash
pip install -r requirements.txt
```

## Instructions
### Datasets
Datasets Tox21, Proteins_full (Proteins with 29 features) will be downloaded automatically. 

Files **Mnet1.pconn.nii** and **netmats1.txt**  for 200 dimensions for HCP dataset should be downloaded from [humanconnectome](https://db.humanconnectome.org/app/template/Login.vm) and added to folder: **data/HCP_PTN1200**

### Execution 
To run the experiments with LGL model
```bash
python main.py --dataset <dataset_name> 
```
Where ```<dataset_name>``` is one of the datasets ```HCP, Tox21, PROTEINS_full```

To run experiments with LGL+KL model
```bash
python main.py --dataset <dataset_name> --regularization "KL_degree_dist"
```
To run experiments with DGCNN model

```bash
python main.py --dataset <dataset_name> --DGCNN True
```
To run experiments with GCN  model 
```bash
python main.py --dataset <dataset_name> --model <model>
```
To run experiments with fixed graphs model
```bash
python main.py --dataset <dataset_name> --model <model> --fix_population_graph <fixed_type>
```
Where ```<model>``` in ``` GCN, FixedGraphInGraph ``` and ```<fixed_type>``` in ```random, knn``` 
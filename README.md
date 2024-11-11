# Introduction

This is the implementation of our paper: [FedL2G: Learning to Guide Local Training in Heterogeneous Federated Learning](https://arxiv.org/abs/2410.06490). 


# Datasets and Environments

Due to the file size limitation, we only upload the statistics (`config.json`) of the Cifar10 dataset in the Dirichlet setting. All the code for the baselines, datasets, and environments is publicly available in the popular repository [HtFLlib](https://github.com/TsingZ0/HtFLlib). 


# System

- `main.py`: System configurations. 
- `total.sh`: Command lines to run experiments for FedL2G with default hyperparameter settings on Linux. 
- `flcore/`: 
    - `clients/`: The code on clients for both FedL2G-l and FedL2G-f. 
    - `servers/`: The code on servers for both FedL2G-l and FedL2G-f. 
    - `trainmodel/`: The code for some heterogeneous client models. 
- `utils/`:
    - `data_utils.py`: The code to read the dataset. 
    - `mem_utils.py`: The code to record memory usage. 
    - `result_utils.py`: The code to save results to files. 

# Training and Evaluation

All codes are stored in `./system`. Just run the following commands.

```
cd ./system
sh run_me.sh
```

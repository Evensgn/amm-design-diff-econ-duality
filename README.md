Designing automated market makers with differentiable economics and strong duality.

## Requirements

* The code for neural network training uses [W&B](https://wandb.ai/site/) for logging, which requires a W&B account.
* The code for numerically solving the dual problem uses [Gurobi solver](https://www.gurobi.com/), which requires a Gurobi license.

## How to use

Create a conda environment
```
conda create -n amm python=3.11
```

Activate the conda environment
```
conda activate amm
```

Install required libraries
```
pip install -r requirements.txt
```

Log into wandb
```
wandb login
```

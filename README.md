# DVF: Multi-agent Q-learning with difference value factorization

This repository is the implementation of the paper "VDF: Multi-agent Q-learning with difference value factorization"[[PDF](https://doi.org/10.1016/j.knosys.2024.111422)]. 

We conduct the experiments using version SC2.4.6.2.69232, which is same as the SMAC run data release (https://github.com/oxwhirl/smac/releases/tag/v1).

## Management log
We use Sacred and Omniboard to manage our results, with the data stored in MongoDB.


## Installation
Set up the Sacred:

```bash
pip install sacred
```

Set up StarCraft II and SMAC with the following command:

```bash
bash install_sc2.sh
```
It will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You also need to set the global environment variable:

```bash
export SC2PATH=[Your SC2 Path/StarCraftII]
```

Install Python environment with command:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install numpy scipy pyyaml pygame pytest probscale imageio snakeviz 
```

## Training
To train DVF in smac , run this command:

```bash
python DVF/main.py --seed with round=1 env_args.map_name=3s5z
```

You can increase parameters with this form:

```bash
python DVF/main.py --seed with round=1 env_args.map_name=3s5z seed=123 
```

All configuration files are placed in config, where `common.yaml` contains some common parameters. Algorithm hyperparameters are placed in `algs/dvf.yaml`. Smac related parameters are placed in `envs/sc2.yaml`. Model and experimental results are stored in the `models` and `results`.

## Ciation

If you found DVF useful, please consider citing with:
```
@article{HUANG2024111422,
title = {DVF:Multi-agent Q-learning with difference value factorization},
journal = {Knowledge-Based Systems},
volume = {286},
pages = {111422},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2024.111422},
author = {Anqi Huang and Yongli Wang and Jianghui Sang and Xiaoli Wang and Yupeng Wang},
}
```


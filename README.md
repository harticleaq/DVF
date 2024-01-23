# VDF: Multi-agent Q-learning with difference value factorization

This repository is the implementation of the paper "VDF: Multi-agent Q-learning with difference value factorization"[[PDF](https://doi.org/10.1016/j.knosys.2024.111422)]. 

We do the experiments in the version of SC2.4.6.2.69232, which is same as the SMAC run data release (https://github.com/oxwhirl/smac/releases/tag/v1).

## Management log
We use Sacred and Omniboard to manage our results, the data is saved in MongoDB. 


## Installation
Set up the Sacred:

```bash
pip install sacred
```

Set up StarCraft II and SMAC with the following command:

```bash
export SC2PATH=[Your SC2 Path/StarCraftII]
```

It will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over.

Install Python environment with command:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install numpy scipy pyyaml pygame pytest probscale imageio snakeviz 
```

## Training
To train DVF in smac , run this command:
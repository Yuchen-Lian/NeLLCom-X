# NeLLCom-X: A Comprehensive Neural-Agent Framework to Simulate Language Learning and Group Communication

![GitHub](https://img.shields.io/github/license/facebookresearch/EGG)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Introduction

NeLLCom-X is a framework that allows researchers to quickly implement multi-agent miniature language learning games. 

The implementation of NeLLCom-X is partly based on EGG toolkit and NeLLCom framework.

More details can be found in our arxiv paper, 
titled "NeLLCom-X: A Comprehensive Neural-Agent Framework to Simulate Language Learning and Group Communication":
[arxiv](https://arxiv.org/*)

## List of Explored Language Features

* Word-order/Case-marking Trade-off

## Agents Architecture

Speaking Agent
* Encoder: Linear
* Decoder: GRU

Listening Agent
* Encoder: GRU
* Decoder: Linear


## Installing NeLLCom-X

Generally, we assume that you use PyTorch 1.1.0 or newer and Python 3.6 or newer.

1. Installing [EGG](https://github.com/facebookresearch/EGG.git.) toolkit;
2. Moving to the EGG game design folder:
   ```
   cd EGG/egg/zoo
   ```
3. Cloning the NeLLCom-X into the EGG game design folder:
   ```
   git clone git@github.com:Yuchen-Lian/NeLLCom-X.git
   cd NeLLCom-X
   ```
4. Then, we can run a game, e.g. the Word-order/Case-marking trade-off game:
    ```bash
    python -m egg.zoo.nellcom-x.train --n_rounds=100
    ```

## NeLLCom-X structure

* `data/` and `data_expand/` contains the full dataset of the predefined artificial languages that are used in the paper.
* `train.py` contain the actual logic implementation.
* `games_*.py` contain the communication pipeline of the game.
* `archs_*.py` contain the agent stucture design.
* `pytorch-seq2seq/` is a git submodule containing a 3rd party seq2seq [framework](https://github.com/IBM/pytorch-seq2seq/).


## Citation
If you find NeLLCom-X useful in your research, please cite this paper:
```
@article{lian2024nellcomx,
  title={NeLLCom-X: A Comprehensive Neural-Agent Framework to Simulate Language Learning and Group Communication},
  author={Lian, Yuchen and Verhoef, Tessa and Bisazza, Arianna},
  journal={arXiv preprint arXiv:*},
  year={2023}
}
```

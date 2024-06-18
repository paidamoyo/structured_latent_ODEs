# Capturing Actionable Dynamics with Structured Latent Ordinary Differential Equations (UAI 2022)

This repository contains the Pytorch code to replicate experiments in our paper [Capturing Actionable Dynamics with Structured Latent Ordinary Differential Equations](https://proceedings.mlr.press/v180/chapfuwa22a/chapfuwa22a.pdf) accepted at Conference on Uncertainty in Artificial Intelligence (UAI 2022):

```latex
@inproceedings{chapfuwa2022capturing,
  title={Capturing Actionable Dynamics with Structured Latent Ordinary Differential Equations},
  author={Chapfuwa, Paidamoyo and Rose, Sherri and Carin, Lawrence and Meeds, Edward and Henao, Ricardo},
  booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
  year={2022}
}
```

## Model
![Model](figures/model.png)

## Prerequisites

The code is implemented with the following dependencies:

- [Python  3.8.10](https://github.com/pyenv/pyenv)
- [Torch 2.3.0](https://pytorch.org/)
- [Pyro 1.9.0](https://pyro.ai/)
- Additional python packages can be installed by running: 

```
pip install -r requirements.txt
```

## Data
We consider the following datasets:
- [Human Viral Challenge](https://gitlab.eecs.umich.edu/yayazhai/shezhai_bme2020)
- [Cardiovascular System (CVS)](cvs.ipynb) 
- [Synthetic Biology](https://github.com/microsoft/vi-hds)

## Model Training

* To train the data specific **SLODE** models run:
  - [training_challenge.py](training_challenge.py)
  - [training_cvs.py](training_cvs.py) 
  - [training_proc.py](training_proc.py)

* The data specific hyper-parameters settings can be found at:
  - [config_challenge.py](data/challenge/config_challenge.py)
  - [config_cvs.py](data/cvs/config_cvs.py) 
  - [config_proc.py](data/proc/config_proc.py)

## Metrics and Visualizations

* Once the networks are trained and the results are saved, we visualize the data specific key results:
  - [challenge_eval_folds.ipynb](challenge_eval_folds.ipynb) for cross validation experiments and [challenge_eval_folds_subject_final.ipynb](challenge_eval_folds_subject_final.ipynb) for subject specific qualitative results
  - [cvs_eval_final.ipynb](cvs_eval_final.ipynb)
  - [sbio_eval_folds_final.ipynb](sbio_eval_folds_final.ipynb) for cross validation experiments and [sbio_eval_heldout_final.ipynb](sbio_eval_heldout_final.ipynb) for zero-shot heldout device

# Weak Supervision Performance Evaluation via Partial Identification

Welcome to our GitHub repository! Here you will find more information about our method of estimating upper and lower bounds for the performance of weak supervision models introduced in

[Maia Polo, Felipe, Subha Maity, Mikhail Yurochkin, Moulinath Banerjee, and Yuekai Sun. "Weak Supervision Performance Evaluation via Partial Identification." In The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024).](https://openreview.net/forum?id=VOVyeOzZx0)

## Overview

Programmatic Weak Supervision (PWS) enables supervised model training without direct access to ground truth labels by leveraging weak labels from heuristics, crowdsourcing, or pre-trained models. However, evaluating these models is challenging because traditional metrics such as accuracy, precision, and recall require labeled data. This repository introduces our implementation of a novel method for evaluating weakly supervised models by framing the task as a **partial identification problem**. Using Fréchet bounds, we estimate reliable performance bounds for key metrics—such as accuracy, precision, recall, and F1-score—without requiring labeled data. Our approach leverages scalable convex optimization to compute these bounds efficiently, even in high-dimensional settings. This framework provides a robust and practical solution for assessing model quality in weak supervision scenarios, overcoming core limitations in existing evaluation techniques.

## Installation

To use the code in this repository, clone the repo and create a conda environment using:

```
conda env create --file=wsbounds.yaml
conda activate wsbounds
```

##  Quick start

Please check our [demo](https://github.com/felipemaiapolo/wsbounds/blob/main/notebooks/demo.ipynb) on how to use our method to evaluate a classifier trained using PWS to hate speech detection.


## Reproducing results from the paper

1. Please download [Wrench](https://github.com/JieyuZ2/wrench) data from [wrench_class.zip](https://drive.google.com/file/d/1m0vdbFiLmdL-IlTL6r0ewhAmub2s1Cuo/view?usp=sharing), unzip the folder, and place it inside the data folder. The data in `wrench_class` is processed using `wsbounds/process_data.py` in case you need to re-process it.
2. Run `python experiments.py --exp1 --exp2 --exp3 --exp4` to re-run all experiments and the plots are generated using the notebooks inside the folder `notebooks`. The file `experiments.py` can be found inside the `wsbounds` folder. In case, you need to re-generate the weak labels for the `spam` experiment, please take a look at `wsbounds/generate_weak_labels.py`.


## Citing

```
@inproceedings{
polo2024weak,
title={Weak Supervision Performance Evaluation via Partial Identification},
author={Felipe Maia Polo and Subha Maity and Mikhail Yurochkin and Moulinath Banerjee and Yuekai Sun},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=VOVyeOzZx0}
}
```

# Universal representation learning for multivariate time series using the instance‐level and cluster‐level supervised contrastive learning

This repository contains the PyToch implementation of the SupCon-TS as described in the paper.


![](figs/approach.png)


# Datasets Preparation

Download all the UEA Multivariate time series archive datasets from [here] (http://www.timeseriesclassification.com/) 

The data structure should look like this:

```none
SupCon-TSC
├── ...
├── data
│   ├── BasicMotion
│   ├── Cricket
```

# Overview

This work has proposed supervised contrastive learning for time series classifcation
(SupCon-TSC). This model is based on the instance-level and cluster-level supervised
contrastive learning approaches to learn the discriminative and universal representation for the multivariate time series dataset

![](figs/critical_diagram.png)


# Train
To train the model, please run the command below:

python main.py --lr1 0.001 --lr2 0.001 --batch_size_emb 10 --batch_size_cl 5 --Epoch 100 --name_dataset BasicMotions


# Citation

Moradinasab, N., Sharma, S., Bar-Yoseph, R., Radom-Aizik, S., C Bilchick, K., M Cooper, D., Weltman, A. and Brown, D.E., 2024. Universal representation learning for multivariate time series using the instance-level and cluster-level supervised contrastive learning. Data Mining and Knowledge Discovery, pp.1-27.

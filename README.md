# Recommender System 2021 Challenge
[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi)

This repository contains the code used for the [Recommender System 2021 Challenge](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi) hosted by the Recommender Systems course at [Politecnico di Milano](https://www.polimi.it/).
The repository is split in 2 main folders:
* [Challenge2021](https://github.com/Menta99/RecSys2021_Mainetti_Menta/tree/main/Challenge2021) which contains our custom models and scripts created for the competition
* [RecSysCourseMaterial](https://github.com/Menta99/RecSys2021_Mainetti_Menta/tree/main/RecSysCourseMaterial) which contains the codebase from the course framework repo

## Overview

The complete description of the problem can be found in the [kaggle competition page](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi/overview). 

Briefly, given the **User Rating Matrix** and some **Item Content Matrices**, the objective of the competition was to create a recommender for **TV series/Movies**.

The evaluation metric used was the **MAP@10**.
| <img src="assets/AP@10_formula.png" width="180"/> | <img src="assets/MAP@10_formula.png" width="180"/> |
|:---:|:---:| 


After a preprocessing phase, we used the following dataset:

* **URM**, 
  * **13650** users
  * **18059** items
  * **2.14%** data sparsity
* **ICM**
  * **18059** items 
  * **335** attributes
  * **1.29%** data sparsity

## Strategy
We approached the problem through different stages:
* At first, we performed some **data exploration**, in order to find interesting patterns in the dataset, 
we discovered in fact the following singularities:
  * Some episodes belong to more than one TV series/Movie
  * Some TV series/Movie even if without channel have been seen by some users
  * Some TV series/Movie even if without episodes have been seen by some users
* Then we **profiled** the base models to find the best performers, both in general and in the different 
user segments (**cold**, **warm** and **hot**).
* The next phase was focused on building **hybrids**, mainly composed by **2 models** at time in order 
to better control their optimization.

[Here](https://github.com/Menta99/RecSys2021_Mainetti_Menta/blob/main/Presentation.pdf) a more complete presentation of the steps that we followed towards our best model.

## Best Model

The **ICMs** were not so effective in our experiments, thus we decided to focus on the information contained in the **URM**.
Our best model was in fact a **Collaborative Stratified Hybrid**, composed by different models aggregated at distinct stages.

In particular the final structure was the following:

<p align="center">
	<img src="assets/diagram.jpg" alt="Diagram"/>
</p>

We opted for a hierarchical structure that increasingly improved the performance of each submodel. 
1. We first separately trained and fine-tuned the base models: 
	- **SLIM Elastic-Net**, that reached a MAP of 0.2501 on the validation set
	- **SLIM-BPR**, that was trained on the Cold user segment reaching a MAP of 0.1446 on the validation set
2. Then we built our **MINT_Cold_v2** hybrid, we co-trained two models: 
	- IALS 
	- MINT_KNN_Hybrid, another hybrid made of ItemKNNCF and UserKNNCF
	> The MINT_Cold_v2 was again trained on the Cold user segment reaching a MAP of 0.1604 on the validation set.
3. At this point we created the **Final_Cold_Hybrid** linearly combining the two models trained on the Cold user segment:
	- SLIM_BPR
	- MINT_Cold_v2
	> This model reached a MAP of 0.1684 on the validation set considering only the Cold user segment.
4. Our **Final_Hybrid** was built segmenting the users (the sizes of the user segments are an hyper-parameter of the model) and linearly combining:
	- SLIM Elastic-Net
	- Final_Cold_Hybrid
	> The Final_Hybrid reached a MAP of 0.2575 on the validation set.

### Evaluation
- **Public** Leaderboard score: **0.50910** (2nd)
- **Private** Leaderboard score: **0.50787** (2nd)

## Group Members
- [__Lorenzo Mainetti__](https://github.com/LorenzoMainetti)
- [__Andrea Menta__](https://github.com/Menta99)

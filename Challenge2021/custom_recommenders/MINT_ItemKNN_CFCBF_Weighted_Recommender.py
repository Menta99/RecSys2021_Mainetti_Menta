#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/2021

@author: Andrea Menta
"""
import numpy as np
import scipy.sparse as sps

from RecSysCourseMaterial.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class MINT_ItemKNN_CFCBF_Weighted_Recommender(ItemKNNCBFRecommender):
    RECOMMENDER_NAME = "MINT_ItemKNN_CFCBF_Weighted_Recommender"

    def __init__(self, URM_train, ICM_train, verbose = True):
        self.ICM_train_list = ICM_train
        super(MINT_ItemKNN_CFCBF_Weighted_Recommender, self).__init__(URM_train, sps.hstack(ICM_train), verbose = verbose)

    def fit(self, ICM_genre_weight, ICM_subgenre_weight, ICM_channel_weight, **fit_args):
        ICM_weights = [ICM_genre_weight, ICM_subgenre_weight, ICM_channel_weight]
        self.ICM_train = [self.ICM_train_list[index] * weight for index, weight in enumerate(ICM_weights)]
        self.ICM_train.append(self.URM_train.T)
        self.ICM_train = sps.hstack(self.ICM_train, format='csr')

        super(MINT_ItemKNN_CFCBF_Weighted_Recommender, self).fit(**fit_args)

    def _get_cold_item_mask(self):
        return np.logical_and(self._cold_item_CBF_mask, self._cold_item_mask)

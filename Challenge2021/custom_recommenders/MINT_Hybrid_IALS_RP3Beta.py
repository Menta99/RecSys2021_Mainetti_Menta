#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/12/2021

@author: Andrea Menta
"""

from RecSysCourseMaterial.Recommenders.BaseRecommender import BaseRecommender
from RecSysCourseMaterial.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSysCourseMaterial.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender


class MINT_Hybrid_IALS_RP3Beta(BaseRecommender):
    RECOMMENDER_NAME = "MINT_Hybrid_IALS_RP3Beta"

    def __init__(self, URM_train):
        super(MINT_Hybrid_IALS_RP3Beta, self).__init__(URM_train)

    def fit(self, IALS_epochs, IALS_num_factors, IALS_confidence_scaling, IALS_alpha, IALS_epsilon, IALS_reg,
            IALS_init_mean, IALS_init_std, RP3Beta_topK, RP3Beta_alpha, RP3Beta_beta,RP3Beta_normalize_similarity,
            IALS_weight, RP3Beta_weight, **earlystopping_kwargs):
        # Instantiate the recommenders and the weights
        self.IALS = IALSRecommender(self.URM_train)
        self.RP3Beta = RP3betaRecommender(self.URM_train)

        self.IALS_weight = IALS_weight
        self.RP3Beta_weight = RP3Beta_weight
        self.Weight_sum = IALS_weight + RP3Beta_weight

        # Fit the recommenders
        self.IALS.fit(epochs=IALS_epochs, num_factors=IALS_num_factors,
                      confidence_scaling=IALS_confidence_scaling, alpha=IALS_alpha,
                      epsilon=IALS_epsilon, reg=IALS_reg, init_mean=IALS_init_mean,
                      init_std=IALS_init_std, **earlystopping_kwargs)
        magic = {
            "topK": RP3Beta_topK,
            "alpha": RP3Beta_alpha,
            "beta": RP3Beta_beta,
            "normalize_similarity": RP3Beta_normalize_similarity
        }
        self.RP3Beta.fit(**magic)


    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_IALS = self.IALS._compute_item_score(user_id_array)
        item_weights_RP3Beta = self.RP3Beta._compute_item_score(user_id_array)

        item_weights = ((self.IALS_weight * item_weights_IALS) + (self.RP3Beta_weight * item_weights_RP3Beta)) / self.Weight_sum

        return item_weights
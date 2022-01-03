#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/12/2021

@author: Andrea Menta
"""

from numpy import linalg as LA

from Challenge2021.custom_recommenders.MINT_Hybrid_IALS_RP3Beta import MINT_Hybrid_IALS_RP3Beta
from Challenge2021.custom_recommenders.MINT_KNN_Hybrid import MINT_KNN_Hybrid
from RecSysCourseMaterial.Recommenders.BaseRecommender import BaseRecommender
from RecSysCourseMaterial.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from RecSysCourseMaterial.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from RecSysCourseMaterial.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender


class MINT_Cold_v0(BaseRecommender):
    RECOMMENDER_NAME = "MINT_Cold"

    def __init__(self, URM_train, ICM):
        self.ICM = ICM
        super(MINT_Cold_v0, self).__init__(URM_train)

    def fit(self, ICM_weight, num_factors, w, norm, **fit_args):
        self.w = w
        self.norm = norm
        # Instantiate the recommenders
        self.SVD = PureSVDRecommender(self.URM_train)
        self.KNN = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM)
        # Fit the recommenders
        self.SVD.fit(num_factors=num_factors, random_seed=1234)
        self.KNN.fit(ICM_weight, **fit_args)

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_SVD = self.SVD._compute_item_score(user_id_array)
        item_weights_KNN = self.KNN._compute_item_score(user_id_array)
        if self.norm:
            item_weights_SVD /= LA.norm(item_weights_SVD, 2)
            item_weights_KNN /= LA.norm(item_weights_KNN, 2)

        item_weights = item_weights_SVD * self.w + (1 - self.w) * item_weights_KNN

        return item_weights


class MINT_Cold_v1(BaseRecommender):
    RECOMMENDER_NAME = "MINT_Cold"

    def __init__(self, URM_train, ICM):
        self.ICM = ICM
        super(MINT_Cold_v1, self).__init__(URM_train)

    def fit(self, ICM_weight, topK, shrink, similarity, normalize, feature_weighting,
            epochs, num_factors, alpha, epsilon, reg, init_mean, init_std, w, norm, **early_stopping):
        self.w = w
        self.norm = norm
        # Instantiate the recommenders
        self.IALS = IALSRecommender(self.URM_train)
        self.KNN = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM)
        # Fit the recommenders
        self.IALS.fit(epochs=epochs, num_factors=num_factors, confidence_scaling='log', alpha=alpha,
                      epsilon=epsilon, reg=reg, init_mean=init_mean, init_std=init_std, **early_stopping)
        magic = {"topK": topK,
                 "shrink": shrink,
                 "similarity": similarity,
                 "normalize": normalize,
                 "feature_weighting": feature_weighting}
        self.KNN.fit(ICM_weight, **magic)

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_IALS = self.IALS._compute_item_score(user_id_array)
        item_weights_KNN = self.KNN._compute_item_score(user_id_array)
        if self.norm:
            item_weights_IALS /= LA.norm(item_weights_IALS, 2)
            item_weights_KNN /= LA.norm(item_weights_KNN, 2)

        item_weights = item_weights_IALS * self.w + (1 - self.w) * item_weights_KNN

        return item_weights


class MINT_Cold_v2(BaseRecommender):
    RECOMMENDER_NAME = "MINT_Cold_v2"

    def __init__(self, URM_train):
        super(MINT_Cold_v2, self).__init__(URM_train)

    def fit(self, Item_topK, Item_shrink, Item_similarity, Item_normalize, Item_feature_weighting,
            User_topK, User_shrink, User_similarity, User_normalize, User_feature_weighting, Weight,
            epochs, num_factors, alpha, epsilon, reg, init_mean, init_std, w, norm, **early_stopping):
        self.w = w
        self.norm = norm
        # Instantiate the recommenders
        self.IALS = IALSRecommender(self.URM_train)
        self.KNN = MINT_KNN_Hybrid(self.URM_train)
        # Fit the recommenders
        self.IALS.fit(epochs=epochs, num_factors=num_factors, confidence_scaling='log', alpha=alpha,
                      epsilon=epsilon, reg=reg, init_mean=init_mean, init_std=init_std, **early_stopping)
        magic = {"Item_topK": Item_topK,
                 "Item_shrink": Item_shrink,
                 "Item_similarity": Item_similarity,
                 "Item_normalize": Item_normalize,
                 "Item_feature_weighting": Item_feature_weighting,
                 "User_topK": User_topK,
                 "User_shrink": User_shrink,
                 "User_similarity": User_similarity,
                 "User_normalize": User_normalize,
                 "User_feature_weighting": User_feature_weighting,
                 "Weight": Weight}
        self.KNN.fit(**magic)

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_IALS = self.IALS._compute_item_score(user_id_array)
        item_weights_KNN = self.KNN._compute_item_score(user_id_array, items_to_compute)
        if self.norm:
            item_weights_IALS /= LA.norm(item_weights_IALS, 2)
            item_weights_KNN /= LA.norm(item_weights_KNN, 2)

        item_weights = item_weights_IALS * self.w + (1 - self.w) * item_weights_KNN

        return item_weights


class MINT_Cold_v3(BaseRecommender):
    RECOMMENDER_NAME = "MINT_Cold_v3"

    def __init__(self, URM_train):
        super(MINT_Cold_v3, self).__init__(URM_train)

    def fit(self, Item_topK, Item_shrink, Item_similarity, Item_normalize, Item_feature_weighting,
            User_topK, User_shrink, User_similarity, User_normalize, User_feature_weighting, Weight,
            IALS_epochs, IALS_num_factors, IALS_confidence_scaling, IALS_alpha, IALS_epsilon, IALS_reg, IALS_init_mean,
            IALS_init_std, RP3Beta_topK, RP3Beta_alpha, RP3Beta_beta, RP3Beta_normalize_similarity,
            IALS_weight, RP3Beta_weight, w, norm, **early_stopping):
        self.w = w
        self.norm = norm
        # Instantiate the recommenders
        self.IALS_RP3 = MINT_Hybrid_IALS_RP3Beta(self.URM_train)
        self.KNN = MINT_KNN_Hybrid(self.URM_train)
        # Fit the recommenders
        self.IALS_RP3.fit(IALS_epochs=IALS_epochs, IALS_num_factors=IALS_num_factors,
                          IALS_confidence_scaling=IALS_confidence_scaling,
                          IALS_alpha=IALS_alpha, IALS_epsilon=IALS_epsilon, IALS_reg=IALS_reg,
                          IALS_init_mean=IALS_init_mean,
                          IALS_init_std=IALS_init_std, RP3Beta_topK=RP3Beta_topK, RP3Beta_alpha=RP3Beta_alpha,
                          RP3Beta_beta=RP3Beta_beta, RP3Beta_normalize_similarity=RP3Beta_normalize_similarity,
                          IALS_weight=IALS_weight, RP3Beta_weight=RP3Beta_weight, **early_stopping)
        magic = {"Item_topK": Item_topK,
                 "Item_shrink": Item_shrink,
                 "Item_similarity": Item_similarity,
                 "Item_normalize": Item_normalize,
                 "Item_feature_weighting": Item_feature_weighting,
                 "User_topK": User_topK,
                 "User_shrink": User_shrink,
                 "User_similarity": User_similarity,
                 "User_normalize": User_normalize,
                 "User_feature_weighting": User_feature_weighting,
                 "Weight": Weight}
        self.KNN.fit(**magic)

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_IALS_RP3 = self.IALS_RP3._compute_item_score(user_id_array, items_to_compute)
        item_weights_KNN = self.KNN._compute_item_score(user_id_array, items_to_compute)
        if self.norm:
            item_weights_IALS_RP3 /= LA.norm(item_weights_IALS_RP3, 2)
            item_weights_KNN /= LA.norm(item_weights_KNN, 2)

        item_weights = item_weights_IALS_RP3 * self.w + (1 - self.w) * item_weights_KNN

        return item_weights

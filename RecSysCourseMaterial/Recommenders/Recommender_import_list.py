#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/19

@author: Maurizio Ferrari Dacrema
"""


######################################################################
##########                                                  ##########
##########                  NON PERSONALIZED                ##########
##########                                                  ##########
######################################################################
from RecSysCourseMaterial.Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects



######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from RecSysCourseMaterial.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSysCourseMaterial.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSysCourseMaterial.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysCourseMaterial.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from RecSysCourseMaterial.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from RecSysCourseMaterial.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from RecSysCourseMaterial.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from RecSysCourseMaterial.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from RecSysCourseMaterial.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from RecSysCourseMaterial.Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from RecSysCourseMaterial.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from RecSysCourseMaterial.Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from RecSysCourseMaterial.Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from RecSysCourseMaterial.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from RecSysCourseMaterial.Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from RecSysCourseMaterial.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from RecSysCourseMaterial.Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from RecSysCourseMaterial.Recommenders.FactorizationMachines.LightFMRecommender import LightFMUserHybridRecommender, LightFMItemHybridRecommender

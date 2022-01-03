import json
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import scipy.sparse as sps
from skopt.space import Real, Integer, Categorical

from Challenge2021.custom_recommenders.MINT_Cold import MINT_Cold_v1
from Challenge2021.custom_recommenders.MINT_ScoresHybridRecommender import MINT_ScoresHybridRecommender2
from Challenge2021.custom_recommenders.MINT_ScoresHybridSegmentedRecommender import \
    MINT_ScoresHybridSegmentedRecommender2
from Challenge2021.utils.data_loader import load_matrix_csr, get_user_segmentation, load_ICM_stack_csr, \
    load_ICM_episodes
from RecSysCourseMaterial.Evaluation.Evaluator import EvaluatorHoldout
from RecSysCourseMaterial.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysCourseMaterial.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from RecSysCourseMaterial.Recommenders.DataIO import DataIO
from RecSysCourseMaterial.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysCourseMaterial.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


class ModelTrainer:
    _USER_ZONE = ["all", "cold", "mid", "hot"]

    def __init__(self, num_fold, model_class, hyp_dict, ICM=None,
                 early_params=None, models_list=None, zone="all"):
        self.num_fold = num_fold
        self.models_list = models_list
        self.ICM = ICM
        self.URM_train = sps.load_npz("../saved_models/k_folds/URM_train_" + str(num_fold) + ".npz")
        self.URM_val = sps.load_npz("../saved_models/k_folds/URM_val_" + str(num_fold) + ".npz")

        if zone not in self._USER_ZONE:
            raise ValueError("zone must be one of " + str(self._USER_ZONE))
        self.zone = zone
        if self.zone == "cold":
            self.evaluator_val = get_user_segmentation(self.URM_train, self.URM_val,
                                                       0, int(self.URM_train.shape[0] * 0.2))
        elif self.zone == "mid":
            self.evaluator_val = get_user_segmentation(self.URM_train, self.URM_val,
                                                       int(self.URM_train.shape[0] * 0.2),
                                                       int(self.URM_train.shape[0] * 0.8))
        elif self.zone == "hot":
            self.evaluator_val = get_user_segmentation(self.URM_train, self.URM_val,
                                                       int(self.URM_train.shape[0] * 0.8),
                                                       int(self.URM_train.shape[0]))
        else:
            self.evaluator_val = EvaluatorHoldout(self.URM_val, cutoff_list=[10])

        self.hyp_dict = hyp_dict
        self.recommender_class = model_class
        self.hyp_search = SearchBayesianSkopt(self.recommender_class, evaluator_validation=self.evaluator_val)
        self.args_train = [self.URM_train]

        if self.ICM is not None:
            self.args_train.append(self.ICM)

        if self.models_list is not None:
            self.args_train.extend(self.models_list)

        self.early_params = early_params
        if self.early_params is not None:
            self.early_params = early_params
            self.early_params["evaluator_object"] = self.evaluator_val

        self.rec_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=self.args_train,
            CONSTRUCTOR_KEYWORD_ARGS={},
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS=self.early_params
        )
        self.output_folder_path = "../result_experiments/"
        self.best_hyper_parameters = None
        self.trained_model = None

    def find_hyp(self, num_epochs=100, num_random_starts=40, save_model="no",
                 metric_to_optimize="MAP", cutoff_to_optimize=10):
        self.hyp_search.search(self.rec_input_args,
                               hyperparameter_search_space=self.hyp_dict,
                               n_cases=num_epochs,
                               n_random_starts=num_random_starts,
                               save_model=save_model,
                               output_folder_path=self.output_folder_path,
                               output_file_name_root=self.recommender_class.RECOMMENDER_NAME + "_" + str(self.num_fold),
                               metric_to_optimize=metric_to_optimize,
                               cutoff_to_optimize=cutoff_to_optimize)

        data_loader = DataIO(folder_path=self.output_folder_path)
        search_metadata = data_loader.load_data(
            self.recommender_class.RECOMMENDER_NAME + "_" + str(self.num_fold) + "_metadata.zip")
        self.best_hyper_parameters = search_metadata["hyperparameters_best"]
        if self.early_params is not None:
            self.best_hyper_parameters["epochs"] = self.hyp_search.recommender_instance.epochs_best
        self.create_log(self.best_hyper_parameters)

        self.trained_model = self.recommender_class(*self.args_train)
        self.trained_model.fit(**self.best_hyper_parameters)
        return self.trained_model

    def create_log(self, best_hyp):
        f = open("../saved_models/k_folds/" + self.recommender_class.RECOMMENDER_NAME +
                 "_" + str(self.num_fold) + ".txt", "w")
        f.write(json.dumps(best_hyp))
        f.close()

    def load_params(self, submission=False):
        f = open("../saved_models/k_folds/" + self.recommender_class.RECOMMENDER_NAME +
                 "_" + str(self.num_fold) + ".txt", "r")
        data = f.read()
        self.best_hyper_parameters = json.loads(data)
        f.close()
        if submission:
            URM_all = load_matrix_csr("../de_compressed/data_train.csv", ["UserID", "ItemID", "Interaction"],
                                      matrix_format="csr")
            args_sub = [URM_all]
            if self.ICM is not None:
                args_sub.append(self.ICM)
            if self.models_list is not None:
                args_sub.extend(self.models_list)
            self.trained_model = self.recommender_class(*args_sub)
        else:
            self.trained_model = self.recommender_class(*self.args_train)
        self.trained_model.fit(**self.best_hyper_parameters)
        return self.trained_model


class ParallelOptimizer:
    def __init__(self, num_folds, model_class, hyp_dict, ICM=None, early_params=None,
                 models_list=None, zone="all", num_epochs=100, num_random_starts=40,
                 save_model="no", metric_to_optimize="MAP", cutoff_to_optimize=10):
        self.num_folds = num_folds
        self.model_class = model_class
        self.hyp_dict = hyp_dict
        self.ICM = ICM
        self.early_params = early_params
        self.models_list = models_list
        self.zone = zone
        self.num_epochs = num_epochs
        self.num_random_starts = num_random_starts
        self.save_model = save_model
        self.metric_to_optimize = metric_to_optimize
        self.cutoff_to_optimize = cutoff_to_optimize
        self.pool = ThreadPool()
        self.model_trainer_dict = {}
        self.trained_models_dict = {}
        self.async_results = {}

    def fit(self):
        for k in range(self.num_folds):
            if self.models_list is None:
                self.model_trainer_dict[k] = ModelTrainer(k, self.model_class, self.hyp_dict, self.ICM,
                                                          self.early_params, self.models_list, self.zone)
            else:
                self.model_trainer_dict[k] = ModelTrainer(k, self.model_class, self.hyp_dict, self.ICM,
                                                          self.early_params, self.models_list[k], self.zone)
            self.async_results[k] = self.pool.apply_async(self.model_trainer_dict[k].find_hyp, (self.num_epochs,
                                                                                                self.num_random_starts,
                                                                                                self.save_model,
                                                                                                self.metric_to_optimize,
                                                                                                self.cutoff_to_optimize))
        for k in range(self.num_folds):
            self.trained_models_dict[k] = self.async_results[k].get()


def remove_seen_on_scores(URM, user_id, scores):
    seen = URM.indices[URM.indptr[user_id]:URM.indptr[user_id + 1]]
    scores[seen] = -np.inf
    return scores


def create_submission(URM, models, num_folds, cutoff=10, remove_seen=True,
                      input_path='../de_compressed/data_target_users_test.csv',
                      output_file_name="submission_boost.csv"):
    test_users = pd.read_csv(input_path)
    test_users_ids = test_users['user_id']
    print("Test loaded...")

    weights = {}
    for fold in range(num_folds):
        weights[fold] = models[fold]._compute_item_score(test_users_ids, None)
    scores_batch = sum(weights.values())
    print("Scores computed...")

    for user_index in range(len(test_users_ids)):
        user_id = test_users_ids[user_index]
        if remove_seen:
            scores_batch[user_index, :] = remove_seen_on_scores(URM, user_id, scores_batch[user_index, :])

    relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

    relevant_items_partition_original_value = scores_batch[
        np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
    relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
    ranking = relevant_items_partition[
        np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

    ranking_list = [None] * ranking.shape[0]

    for user_index in range(len(test_users_ids)):
        user_recommendation_list = ranking[user_index]
        user_item_scores = scores_batch[user_index, user_recommendation_list]

        not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

        user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
        ranking_list[user_index] = user_recommendation_list.tolist()

    print("Recommendation produced...")

    test_users['item_list'] = ranking_list
    test_users['item_list'] = pd.DataFrame(
        [str(line).strip('[').strip(']').replace("'", "") for line in test_users['item_list']])

    test_users.to_csv(output_file_name, index=False)


if __name__ == '__main__':
    URM_all = load_matrix_csr("../de_compressed/data_train.csv", ["UserID", "ItemID", "Interaction"], matrix_format="csr")
    ICM_genre_subgenre_channel = load_ICM_stack_csr(
        ["../de_compressed/data_ICM_genre.csv", "../de_compressed/data_ICM_subgenre.csv",
         "../de_compressed/data_ICM_channel.csv"],
        [["ItemID", "GenreID", "Interaction"], ["ItemID", "SubgenreID", "Interaction"],
         ["ItemID", "ChannelID", "Interaction"]], matrix_format="csr")
    ICM_episodes = load_ICM_episodes("csr", True)
    ICM = sps.hstack([ICM_genre_subgenre_channel, ICM_episodes])
    num_folds = 5
    print("Input Data loaded...")

    ###############################################OPTIMIZE SLIME#######################################################

    hyp_dict1 = {
        "alpha": Real(1e-3, 1e-1),
        "l1_ratio": Real(1e-5, 1e-3),
        "topK": Integer(1200, 1300)}
    model1_opt = ParallelOptimizer(num_folds=num_folds, model_class=SLIMElasticNetRecommender, hyp_dict=hyp_dict1,
                                   ICM=None,
                                   early_params=None, models_list=None, zone="all", num_epochs=10, num_random_starts=5,
                                   save_model="no", metric_to_optimize="MAP", cutoff_to_optimize=10)
    model1_opt.fit()
    print("SLIME Finished...")

    ###############################################OPTIMIZE SLIM-BPR####################################################

    hyp_dict2 = {
        "epochs": Categorical([500]),
        "allow_train_with_sparse_weights": Categorical([True, False]),
        "symmetric": Categorical([True, False]),
        "lambda_i": Real(low=1e-3, high=50.0, prior='log-uniform'),
        "lambda_j": Real(low=1e-3, high=50.0, prior='log-uniform'),
        "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform'),
        "topK": Integer(50, 300),
        "sgd_mode": Categorical(['adagrad', 'adadelta']),
        "gamma": Real(0.0, 1.0),
        "beta_1": Real(0.0, 1.0),
        "beta_2": Real(0.0, 1.0)}
    early_params2 = {"validation_every_n": 5,
                     "stop_on_validation": True,
                     "lower_validations_allowed": 5,
                     "validation_metric": "MAP"}
    model2_opt = ParallelOptimizer(num_folds=num_folds, model_class=SLIM_BPR_Cython, hyp_dict=hyp_dict2, ICM=None,
                                   early_params=early_params2, models_list=None, zone="cold", num_epochs=100,
                                   num_random_starts=40, save_model="no", metric_to_optimize="MAP",
                                   cutoff_to_optimize=10)
    model2_opt.fit()
    print("SLIM-BPR Finished...")

    ###############################################OPTIMIZE MINT-COLD###################################################

    hyp_dict3 = {
        'ICM_weight': Real(low=1e-4, high=1, prior='uniform'),
        'topK': Integer(200, 500),
        'shrink': Integer(50, 200),
        'similarity': Categorical(
            ["cosine", "pearson", "jaccard", "tanimoto", "adjusted", "euclidean", "dice", "tversky"]),
        'normalize': Categorical([True, False]),
        'feature_weighting': Categorical(["BM25", "TF-IDF", "none"]),
        'epochs': Categorical([500]),
        'num_factors': Integer(10, 100),
        'alpha': Real(low=1e-2, high=0.5, prior='log-uniform'),
        'epsilon': Real(low=1e-2, high=1e-1, prior='log-uniform'),
        'reg': Real(low=1e-6, high=1e-3, prior='log-uniform'),
        'init_mean': Real(low=1e-3, high=0.5),
        'init_std': Real(low=0.0, high=1),
        'w': Real(low=1e-2, high=1, prior='log-uniform'),
        'norm': Categorical([True, False])}
    early_params3 = {"validation_every_n": 5,
                     "stop_on_validation": True,
                     "lower_validations_allowed": 5,
                     "validation_metric": "MAP"}
    model3_opt = ParallelOptimizer(num_folds=num_folds, model_class=MINT_Cold_v1, hyp_dict=hyp_dict3, ICM=ICM,
                                   early_params=early_params3, models_list=None, zone="cold", num_epochs=100,
                                   num_random_starts=40, save_model="no", metric_to_optimize="MAP",
                                   cutoff_to_optimize=10)
    model3_opt.fit()
    print("MINT-COLD Finished...")

    ###############################################OPTIMIZE FINAL-HYBRID################################################
    
    models_in4 = []
    for k in range(num_folds):
        bpr = ModelTrainer(num_fold=k, model_class=SLIM_BPR_Cython, hyp_dict=None, zone="cold").load_params(submission=False)
        mint_cold = ModelTrainer(num_fold=k, model_class=MINT_Cold_v1, hyp_dict=None, ICM=ICM, zone="cold").load_params(submission=False)
        models_in4.append([bpr, mint_cold])
    hyp_dict4 = {
        "weight_1": Real(low=1e-3, high=1, prior='uniform'),
        "weight_2": Real(low=1e-3, high=1, prior='uniform')}
    model4_opt = ParallelOptimizer(num_folds=num_folds, model_class=MINT_ScoresHybridRecommender2, hyp_dict=hyp_dict4,
                                   ICM=None, early_params=None, models_list=models_in4, zone="all", num_epochs=100,
                                   num_random_starts=40, save_model="no", metric_to_optimize="MAP",
                                   cutoff_to_optimize=10)
    model4_opt.fit()
    print("FINAL-HYBRID Finished...")

    ###############################################OPTIMIZE FINAL#######################################################

    models_in5 = []
    for k in range(num_folds):
        slime = model1_opt.trained_models_dict[k]
        final_hyb = model4_opt.trained_models_dict[k]
        models_in5.append([slime, final_hyb])
    hyp_dict5 = {
        "weight_1": Real(low=1e-3, high=1, prior='uniform'),
        "weight_2": Real(low=1e-3, high=1, prior='uniform'),
        "weight_3": Real(low=1e-3, high=1, prior='uniform')}
    model5_opt = ParallelOptimizer(num_folds=num_folds, model_class=MINT_ScoresHybridSegmentedRecommender2,
                                   hyp_dict=hyp_dict5,
                                   ICM=None, early_params=None, models_list=models_in5, zone="all", num_epochs=100,
                                   num_random_starts=40, save_model="no", metric_to_optimize="MAP",
                                   cutoff_to_optimize=10)
    model5_opt.fit()
    print("FINAL Finished...")

    ###############################################CREATE SUBMISSION####################################################

    slime_all = []
    bpr_all = []
    mint_cold = []
    final_hyb_all = []
    final_all = []
    for k in range(num_folds):
        slime_all.append(model1_opt.model_trainer_dict[k].load_params(submission=True))
        bpr_all.append(model2_opt.model_trainer_dict[k].load_params(submission=True))
        mint_cold.append(model3_opt.model_trainer_dict[k].load_params(submission=True))

        model4_opt.model_trainer_dict[k].models_list = [bpr_all[k], mint_cold[k]]
        final_hyb_all.append(model4_opt.model_trainer_dict[k].load_params(submission=True))

        model5_opt.model_trainer_dict[k].models_list = [slime_all[k], final_hyb_all[k]]
        final_all.append(model5_opt.model_trainer_dict[k].load_params(submission=True))

    create_submission(URM=URM_all, models=final_all, num_folds=num_folds, cutoff=10, remove_seen=True)
    print("Submission created...")


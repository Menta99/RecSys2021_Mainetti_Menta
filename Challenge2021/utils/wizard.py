import os

import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

from Challenge2021.custom_recommenders.MINT_Cold import MINT_Cold_v2
from Challenge2021.custom_recommenders.MINT_ScoresHybridRecommender import MINT_ScoresHybridRecommender2
from Challenge2021.custom_recommenders.MINT_ScoresHybridSegmentedRecommender import \
    MINT_ScoresHybridSegmentedRecommender2
from RecSysCourseMaterial.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from RecSysCourseMaterial.Evaluation.Evaluator import EvaluatorHoldout
from RecSysCourseMaterial.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysCourseMaterial.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from RecSysCourseMaterial.Recommenders.DataIO import DataIO
from RecSysCourseMaterial.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysCourseMaterial.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from data_loader import load_matrix_csr, load_ICM_stack_csr, load_ICM_episodes


def load_input_matrices(load_ICM=True, valid_split=0.8, cutoff=10):
    URM_all = load_matrix_csr("../de_compressed/data_train.csv", ["UserID", "ItemID", "Interaction"],
                              matrix_format="csr")
    print("URM loaded : " + str(URM_all.shape[0]) + "x" + str(URM_all.shape[1]))

    URM_train, URM_val = split_train_in_two_percentage_global_sample(URM_all, train_percentage=valid_split)
    evaluator_val = EvaluatorHoldout(URM_val, cutoff_list=[cutoff])
    input_args = [URM_train]

    ICM = None
    if load_ICM:
        ICM_genre_subgenre_channel = load_ICM_stack_csr(
            ["../de_compressed/data_ICM_genre.csv", "../de_compressed/data_ICM_subgenre.csv",
             "../de_compressed/data_ICM_channel.csv"],
            [["ItemID", "GenreID", "Interaction"], ["ItemID", "SubgenreID", "Interaction"],
             ["ItemID", "ChannelID", "Interaction"]], matrix_format="csr")
        ICM_episodes = load_ICM_episodes("csr", True)
        ICM = sps.hstack([ICM_genre_subgenre_channel, ICM_episodes])
        print("ICM loaded : " + str(ICM.shape[0]) + "x" + str(ICM.shape[1]))
        input_args.append(ICM)

    return input_args, evaluator_val, URM_all, ICM


def find_opt_params(model, hyper_params_dict, num_iterations=100, num_random_steps=50, load_ICM=True, create_sub=False,
                    valid_split=0.8, cutoff=10, metric="MAP", output_folder_path="result_experiments/",
                    **earlystopping_keywargs):
    input_args, evaluator_val, URM_all, ICM = load_input_matrices(load_ICM, valid_split, cutoff)

    hyper_parameter_Search = SearchBayesianSkopt(model, evaluator_validation=evaluator_val)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=input_args,
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS=earlystopping_keywargs
    )

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    print("Parameters set, starts optimization...")

    hyper_parameter_Search.search(recommender_input_args,
                                  hyperparameter_search_space=hyper_params_dict,
                                  n_cases=num_iterations,
                                  n_random_starts=num_random_steps,
                                  save_model="no",
                                  output_folder_path=output_folder_path,
                                  output_file_name_root=model.RECOMMENDER_NAME,
                                  metric_to_optimize=metric,
                                  cutoff_to_optimize=cutoff)

    if create_sub:
        data_loader = DataIO(folder_path=output_folder_path)
        search_metadata = data_loader.load_data(model.RECOMMENDER_NAME + "_metadata.zip")
        best_hyper_parameters = search_metadata["hyperparameters_best"]

        if load_ICM:
            final = model(URM_all, ICM)
        else:
            final = model(URM_all)
        final.fit(**best_hyper_parameters)
        create_submission(final)


def create_submission(model, cutoff=10, remove_seen=True, input_path='../de_compressed/data_target_users_test.csv',
                      output_file_name="submission_final.csv"):
    test_users = pd.read_csv(input_path)
    test_users_ids = test_users['user_id']

    output = [np.array(model.recommend(user, cutoff=cutoff, remove_seen_flag=remove_seen)) for user in
              tqdm(test_users_ids)]
    print("Recommendation produced...")

    test_users['item_list'] = output
    test_users['item_list'] = pd.DataFrame(
        [str(line).strip('[').strip(']').replace("'", "") for line in test_users['item_list']])

    test_users.to_csv(output_file_name, index=False)


if __name__ == '__main__':
    URM_all = load_matrix_csr("../de_compressed/data_train.csv", ["UserID", "ItemID", "Interaction"],
                              matrix_format="csr")
    print("URM loaded : " + str(URM_all.shape[0]) + "x" + str(URM_all.shape[1]))

    model1 = SLIMElasticNetRecommender(URM_all)
    model1.load_model(folder_path="../saved_models/split/", file_name="SLIM_2501_ALL.zip")

    model2 = SLIM_BPR_Cython(URM_all)
    model2.load_model(folder_path="../saved_models/split/", file_name="BPR_COLD_ALL.zip")

    model3 = MINT_Cold_v2(URM_all)
    hyp_mint_cold = {'Item_topK': 190, 'Item_shrink': 168, 'Item_similarity': 'jaccard', 'Item_normalize': True,
                     'Item_feature_weighting': 'none', 'User_topK': 5, 'User_shrink': 33, 'User_similarity': 'jaccard',
                     'User_normalize': True, 'User_feature_weighting': 'TF-IDF', 'Weight': 1e-05, 'epochs': 35,
                     'num_factors': 87, 'alpha': 0.38577317688228835, 'epsilon': 0.01491395933802407, 'reg': 0.001,
                     'init_mean': 0.20190781153520893, 'init_std': 0.6135331047430492, 'w': 0.5301757847758689,
                     'norm': True}
    model3.fit(**hyp_mint_cold)

    model4 = MINT_ScoresHybridRecommender2(URM_all, model2, model3)
    model4.fit(normalize=True, alpha=0.5534172771551772)

    final = MINT_ScoresHybridSegmentedRecommender2(URM_all, model1, model4)
    best_hyp = {'weight_1': 0.6132358257101167, 'weight_2': 0.17496812747035034, 'weight_3': 0.2550008435270654,
                'first_step': 72, 'second_step': 1558}
    final.fit(**best_hyp)
    print("Recommender prepared...")
    create_submission(final)

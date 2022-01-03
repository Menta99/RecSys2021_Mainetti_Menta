import json

import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

from Challenge2021.custom_recommenders.MINT_Cold import MINT_Cold_v1
from Challenge2021.custom_recommenders.MINT_Ensembler import MINT_Ensembler
from Challenge2021.custom_recommenders.MINT_ScoresHybridRecommender import MINT_ScoresHybridRecommender2
from Challenge2021.custom_recommenders.MINT_ScoresHybridSegmentedRecommender import \
    MINT_ScoresHybridSegmentedRecommender2
from Challenge2021.utils.data_loader import load_ICM_stack_csr, load_ICM_episodes, load_matrix_csr
from RecSysCourseMaterial.Evaluation.Evaluator import EvaluatorHoldout
from RecSysCourseMaterial.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysCourseMaterial.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


def load_hyp(model, fold):
    f = open("../saved_models/k_folds/" + model.RECOMMENDER_NAME + "_" + str(fold) + ".txt", "r")
    data = f.read()
    model_hyp = json.loads(data)
    f.close()
    return model_hyp


def create_submission(model, cutoff=10, remove_seen=True, input_path='../de_compressed/data_target_users_test.csv',
                      output_file_name="submission_FireInTheHole_ALL.csv"):
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
    ICM_genre_subgenre_channel = load_ICM_stack_csr(
        ["../de_compressed/data_ICM_genre.csv", "../de_compressed/data_ICM_subgenre.csv",
         "../de_compressed/data_ICM_channel.csv"],
        [["ItemID", "GenreID", "Interaction"], ["ItemID", "SubgenreID", "Interaction"],
         ["ItemID", "ChannelID", "Interaction"]], matrix_format="csr")
    ICM_episodes = load_ICM_episodes("csr", True)
    ICM = sps.hstack([ICM_genre_subgenre_channel, ICM_episodes])

    train_all = True
    URM_train_list = []
    URM_val_list = []
    evaluator_val_list = []
    model1_list = []
    model2_list = []
    model3_list = []
    model4_list = []
    model5_list = []
    folds = 5

    for num_fold in range(folds):
        if train_all:
            URM_train = load_matrix_csr("../de_compressed/data_train.csv", ["UserID", "ItemID", "Interaction"],
                                        matrix_format="csr")
        else:
            URM_train = sps.load_npz("../saved_models/k_folds/URM_train_" + str(num_fold) + ".npz")
            URM_train_list.append(URM_train)
            URM_val = sps.load_npz("../saved_models/k_folds/URM_val_" + str(num_fold) + ".npz")
            URM_val_list.append(URM_val)
            evaluator_val = EvaluatorHoldout(URM_val, cutoff_list=[10])
            evaluator_val_list.append(evaluator_val)

        model1_hyp = load_hyp(SLIMElasticNetRecommender, num_fold)
        model2_hyp = load_hyp(SLIM_BPR_Cython, num_fold)
        model3_hyp = load_hyp(MINT_Cold_v1, num_fold)
        model4_hyp = load_hyp(MINT_ScoresHybridRecommender2, num_fold)
        model5_hyp = load_hyp(MINT_ScoresHybridSegmentedRecommender2, num_fold)

        if train_all:
            model1 = SLIMElasticNetRecommender(URM_train)
            model1.fit(**model1_hyp)
            model1.save_model(folder_path="../saved_models/k_folds/", file_name="SLIM_" + str(num_fold)) + "_ALL"
            model1_list.append(model1)

            model2 = SLIM_BPR_Cython(URM_train)
            model2.fit(**model2_hyp)
            model2.save_model(folder_path="../saved_models/k_folds/", file_name="BPR_" + str(num_fold)) + "_ALL"
            model2_list.append(model2)
        else:
            model1 = SLIMElasticNetRecommender(URM_train)
            model1.load_model(folder_path="../saved_models/k_folds/",
                              file_name="SLIM_" + str(num_fold) + ".zip")
            model1_list.append(model1)

            model2 = SLIM_BPR_Cython(URM_train)
            model2.load_model(folder_path="../saved_models/k_folds/",
                              file_name="BPR_" + str(num_fold) + ".zip")
            model2_list.append(model2)

        model3 = MINT_Cold_v1(URM_train, ICM)
        model3.fit(**model3_hyp)
        model3_list.append(model3)

        model4 = MINT_ScoresHybridRecommender2(URM_train, model2, model3)
        model4.fit(**model4_hyp)
        model4_list.append(model4)

        model5 = MINT_ScoresHybridSegmentedRecommender2(URM_train, model1, model4)
        model5.fit(**model5_hyp)
        model5_list.append(model5)

    if train_all:
        URM_train = load_matrix_csr("../de_compressed/data_train.csv", ["UserID", "ItemID", "Interaction"],
                                    matrix_format="csr")
    else:
        URM_train = sps.load_npz("../saved_models/SLIM_2489/URM_train.npz")

    best_hyper_parameters = {'weight_0': 0.006323135352478189, 'weight_1': 0.03606793002595008,
                             'weight_2': 0.31137294555604245, 'weight_3': 0.12784829135461706,
                             'weight_4': 0.3256546115959311}
    ensembler = MINT_Ensembler(URM_train, model5_list[0], model5_list[1], model5_list[2], model5_list[3],
                               model5_list[4])
    ensembler.fit(**best_hyper_parameters)
    create_submission(ensembler)

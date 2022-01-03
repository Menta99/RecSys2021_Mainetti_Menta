import os

import scipy.sparse as sps
from skopt.space import Real, Categorical

from Challenge2021.custom_recommenders.MINT_Cold import MINT_Cold_v2
from Challenge2021.custom_recommenders.MINT_ScoresHybridRecommender import MINT_ScoresHybridRecommender2
from Challenge2021.utils.data_loader import get_user_segmentation
from RecSysCourseMaterial.Evaluation.Evaluator import EvaluatorHoldout
from RecSysCourseMaterial.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysCourseMaterial.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from RecSysCourseMaterial.Recommenders.DataIO import DataIO
from RecSysCourseMaterial.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

URM_train = sps.load_npz("../saved_models/split/URM_train.npz")
URM_val = sps.load_npz("../saved_models/split/URM_val.npz")
evaluator_val = get_user_segmentation(URM_train, URM_val, 0, int(URM_train.shape[0] * 0.2))
evaluator_val_tot = EvaluatorHoldout(URM_val, cutoff_list=[10])

model_1 = SLIM_BPR_Cython(URM_train)
model_2 = MINT_Cold_v2(URM_train)

model_1.load_model(folder_path="../saved_models/split/", file_name="BPR_COLD.zip")

hyp_mint_cold_v2 = {'Item_topK': 190, 'Item_shrink': 168, 'Item_similarity': 'jaccard', 'Item_normalize': True,
                    'Item_feature_weighting': 'none', 'User_topK': 5, 'User_shrink': 33, 'User_similarity': 'jaccard',
                    'User_normalize': True, 'User_feature_weighting': 'TF-IDF', 'Weight': 1e-05, 'epochs': 35,
                    'num_factors': 87, 'alpha': 0.38577317688228835, 'epsilon': 0.01491395933802407, 'reg': 0.001,
                    'init_mean': 0.20190781153520893, 'init_std': 0.6135331047430492, 'w': 0.5301757847758689,
                    'norm': True}

model_2.fit(**hyp_mint_cold_v2)

print("Models prepared")
print("Models performances: \nSLIM_BPR: " + str(evaluator_val.evaluateRecommender(model_1)[0]["MAP"][10])
      + "\nMINT_Cold_IALS_v2: " + str(evaluator_val.evaluateRecommender(model_2)[0]["MAP"][10]))
print("Models performances: \nSLIM_BPR_ALL: " + str(evaluator_val_tot.evaluateRecommender(model_1)[0]["MAP"][10])
      + "\nMINT_Cold_IALS_v2_ALL: " + str(evaluator_val_tot.evaluateRecommender(model_2)[0]["MAP"][10]))

hyper_parameters_range_dictionary = {
    "alpha": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "normalize": Categorical([True, False])
}

recommender_class = MINT_ScoresHybridRecommender2

hyper_parameter_Search = SearchBayesianSkopt(recommender_class,
                                             evaluator_validation=evaluator_val)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, model_1, model_2],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={}
)

output_folder_path = "../result_experiments/"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

print("Parameters set, starts optimization...")

hyper_parameter_Search.search(recommender_input_args,
                              hyperparameter_search_space=hyper_parameters_range_dictionary,
                              n_cases=100,
                              n_random_starts=40,
                              save_model="no",
                              output_folder_path=output_folder_path,
                              output_file_name_root=recommender_class.RECOMMENDER_NAME,
                              metric_to_optimize="MAP",
                              cutoff_to_optimize=10)

data_loader = DataIO(folder_path=output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

hyper_parameters_df = search_metadata["hyperparameters_df"]
print(hyper_parameters_df)

result_on_validation_df = search_metadata["result_on_validation_df"]
print(result_on_validation_df)

best_hyper_parameters = search_metadata["hyperparameters_best"]
print(best_hyper_parameters)

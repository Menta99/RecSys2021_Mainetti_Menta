import os

import scipy.sparse as sps
from skopt.space import Real, Integer

from Challenge2021.custom_recommenders.MINT_Cold import MINT_Cold_v2
from Challenge2021.custom_recommenders.MINT_ScoresHybridRecommender import MINT_ScoresHybridRecommender2
from Challenge2021.custom_recommenders.MINT_ScoresHybridSegmentedRecommender import \
    MINT_ScoresHybridSegmentedRecommender2
from RecSysCourseMaterial.Evaluation.Evaluator import EvaluatorHoldout
from RecSysCourseMaterial.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysCourseMaterial.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from RecSysCourseMaterial.Recommenders.DataIO import DataIO
from RecSysCourseMaterial.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysCourseMaterial.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

URM_train = sps.load_npz("../saved_models/split/URM_train.npz")
URM_val = sps.load_npz("../saved_models/split/URM_val.npz")
evaluator_val = EvaluatorHoldout(URM_val, cutoff_list=[10])

model1 = SLIMElasticNetRecommender(URM_train)
model1.load_model(folder_path="../saved_models/split/", file_name="SLIM_2501.zip")

model2 = SLIM_BPR_Cython(URM_train)
model2.load_model(folder_path="../saved_models/split/", file_name="BPR_COLD.zip")

model3 = MINT_Cold_v2(URM_train)

hyp_mint_cold = {'Item_topK': 190, 'Item_shrink': 168, 'Item_similarity': 'jaccard', 'Item_normalize': True,
                 'Item_feature_weighting': 'none', 'User_topK': 5, 'User_shrink': 33, 'User_similarity': 'jaccard',
                 'User_normalize': True, 'User_feature_weighting': 'TF-IDF', 'Weight': 1e-05, 'epochs': 35,
                 'num_factors': 87, 'alpha': 0.38577317688228835, 'epsilon': 0.01491395933802407, 'reg': 0.001,
                 'init_mean': 0.20190781153520893, 'init_std': 0.6135331047430492, 'w': 0.5301757847758689, 'norm': True}

model3.fit(**hyp_mint_cold)

model4 = MINT_ScoresHybridRecommender2(URM_train, model2, model3)
model4.fit(normalize=True, alpha=0.5534172771551772)

print("Models loaded...")
print("Models performances: \nSLIM: " + str(evaluator_val.evaluateRecommender(model1)[0]["MAP"][10])
      + "\nFINAL_Cold: " + str(evaluator_val.evaluateRecommender(model4)[0]["MAP"][10]))

hyper_parameters_range_dictionary = {
    "weight_1": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "weight_2": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "weight_3": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "first_step": Integer(10, 200),
    "second_step": Integer(500, 1800)
}

recommender_class = MINT_ScoresHybridSegmentedRecommender2

hyper_parameter_Search = SearchBayesianSkopt(recommender_class,
                                             evaluator_validation=evaluator_val)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, model1, model4],
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
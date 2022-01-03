import os

import scipy.sparse as sps
from skopt.space import Real, Integer

from RecSysCourseMaterial.Evaluation.Evaluator import EvaluatorHoldout
from RecSysCourseMaterial.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysCourseMaterial.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from RecSysCourseMaterial.Recommenders.DataIO import DataIO
from RecSysCourseMaterial.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

URM_train = sps.load_npz("../saved_models/split/URM_train.npz")
URM_val = sps.load_npz("../saved_models/split/URM_val.npz")
evaluator_val = EvaluatorHoldout(URM_val, cutoff_list=[10])

hyper_parameters_range_dictionary = {
    "alpha": Real(1e-3, 1e-1),
    "l1_ratio": Real(1e-5, 1e-3),
    "topK": Integer(1100, 1300)
}

recommender_class = SLIMElasticNetRecommender

hyper_parameter_Search = SearchBayesianSkopt(recommender_class,
                                             evaluator_validation=evaluator_val)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
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

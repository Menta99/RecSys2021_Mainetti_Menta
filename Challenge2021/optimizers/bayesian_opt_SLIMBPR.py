import os

import scipy.sparse as sps
from skopt.space import Real, Integer, Categorical

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

hyper_parameters_range_dictionary = {
    "epochs": Categorical([500]),
    "allow_train_with_sparse_weights": Categorical([True, False]),
    "symmetric": Categorical([True, False]),
    "lambda_i": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "lambda_j": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform'),
    "topK": Integer(100, 300),
    "sgd_mode": Categorical(['adagrad', 'adadelta']),
    "gamma": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "beta_1": Real(low=1e-3, high=1.0, prior='log-uniform'),
    "beta_2": Real(low=1e-3, high=1.0, prior='log-uniform')
}

earlystopping_keywargs = {"validation_every_n": 5,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_val,
                          "lower_validations_allowed": 5,
                          "validation_metric": "MAP",
                          }

recommender_class = SLIM_BPR_Cython

hyper_parameter_Search = SearchBayesianSkopt(recommender_class,
                                             evaluator_validation=evaluator_val)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS=earlystopping_keywargs
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

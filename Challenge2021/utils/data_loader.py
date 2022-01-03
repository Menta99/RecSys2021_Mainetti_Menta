import numpy as np
import pandas as pd
import scipy.sparse as sps

from RecSysCourseMaterial.Evaluation.Evaluator import EvaluatorHoldout


def load_matrix_csr(matrix_path, columns, matrix_format="csr"):
    matrix_df = pd.read_csv(filepath_or_buffer=matrix_path,
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: float},
                            engine='python')

    matrix_df.columns = columns
    matrix = sps.coo_matrix((matrix_df[columns[2]].values,
                             (matrix_df[columns[0]].values, matrix_df[columns[1]].values)))
    if matrix_format == "csr":
        return matrix.tocsr()
    else:
        return matrix.tocsc()


def load_ICM_stack_csr(ICM_path_list, columns_list, matrix_format="csr"):
    assert len(ICM_path_list) == len(
        columns_list), "list of columns and list of path of different dimensions, required same"

    dataframes = []
    for index in range(len(ICM_path_list)):
        matrix_df = pd.read_csv(filepath_or_buffer=ICM_path_list[index],
                                sep=",",
                                skiprows=1,
                                header=None,
                                dtype={0: int, 1: int, 2: float},
                                engine='python')

        matrix_df.columns = columns_list[index]
        class_num = matrix_df[columns_list[index][1]].nunique()
        matrix = sps.coo_matrix((matrix_df[columns_list[index][2]].values,
                                 (matrix_df[columns_list[index][0]].values, matrix_df[columns_list[index][1]].values)))

        dataframes.append(pd.DataFrame(data=matrix.todense(),
                                       columns=[str(i) + "° " + columns_list[index][1] for i in range(class_num)]))

    if matrix_format == "csr":
        return sps.csr_matrix(pd.concat(dataframes, axis=1).to_numpy())
    else:
        return sps.csc_matrix(pd.concat(dataframes, axis=1).to_numpy())


def load_ICM_episodes(matrix_format="csr", clean=True):
    matrix_df = pd.read_csv(filepath_or_buffer="../de_compressed/data_ICM_event.csv",
                            sep=",",
                            skiprows=1,
                            header=None,
                            dtype={0: int, 1: int, 2: float},
                            engine='python')
    matrix_df.columns = ['Item_id', 'Episode_id', 'Interaction']

    if clean:
        matrix_df = matrix_df.drop_duplicates(subset='Episode_id', keep=False)

    item_episodes = np.zeros(matrix_df["Item_id"].max() + 1, dtype=int)
    for index, row in matrix_df.iterrows():
        item_episodes[int(row["Item_id"])] += 1

    if matrix_format == "csr":
        return sps.csr_matrix(pd.DataFrame(data=item_episodes, columns=["Length"]).to_numpy())
    else:
        return sps.csc_matrix(pd.DataFrame(data=item_episodes, columns=["Length"]).to_numpy())


def get_user_segmentation(URM_train, URM_val, start_pos, end_pos):
    profile_length = np.ediff1d(URM_train.indptr)
    sorted_users = np.argsort(profile_length)
    users_in_group = sorted_users[start_pos:end_pos]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    return EvaluatorHoldout(URM_val, cutoff_list=[10], ignore_users=users_not_in_group)

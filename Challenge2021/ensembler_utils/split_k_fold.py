import numpy as np
import scipy.sparse as sps

from RecSysCourseMaterial.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


def split_k_fold_global_sample(URM_all, number_of_folds=10):
    """
    The function creates a dictionary composed by the k-folds of the matrix (used for validation, position 1) and their
    complementary (used for training, position 0), using as key the index of the fold
    :param URM_all:
    :param number_of_folds:
    :return:
    """
    assert type(number_of_folds) == int, "number_of_folds must be an integer"

    num_users, num_items = URM_all.shape
    URM_train = sps.coo_matrix(URM_all)

    indices_for_sampling = np.arange(0, URM_all.nnz, dtype=np.int)
    np.random.shuffle(indices_for_sampling)

    n_interactions_per_fold = int(np.floor(URM_all.nnz / number_of_folds))

    folds = {}
    for k in range(number_of_folds):
        indices_for_validation = indices_for_sampling[k * n_interactions_per_fold: (k + 1) * n_interactions_per_fold]
        # indices_for_train = np.array([i for i in indices_for_sampling if i not in indices_for_validation])
        indices_for_train = np.concatenate((indices_for_sampling[:k * n_interactions_per_fold],
                                            indices_for_sampling[(k + 1) * n_interactions_per_fold:]), axis=None)
        if k == number_of_folds - 1:
            indices_for_validation = indices_for_sampling[k * n_interactions_per_fold:]
            indices_for_train = indices_for_sampling[:k * n_interactions_per_fold]

        # assert np.array_equal(np.sort(indices_for_sampling),np.sort(np.concatenate((indices_for_train,
        # indices_for_validation), axis=None)))

        URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items, auto_create_col_mapper=False,
                                                    auto_create_row_mapper=False)
        URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items,
                                                         auto_create_col_mapper=False, auto_create_row_mapper=False)

        URM_train_builder.add_data_lists(URM_train.row[indices_for_train],
                                         URM_train.col[indices_for_train],
                                         URM_train.data[indices_for_train])

        URM_validation_builder.add_data_lists(URM_train.row[indices_for_validation],
                                              URM_train.col[indices_for_validation],
                                              URM_train.data[indices_for_validation])

        URM_train_final = sps.csr_matrix(URM_train_builder.get_SparseMatrix())
        URM_validation_final = sps.csr_matrix(URM_validation_builder.get_SparseMatrix())

        user_no_item_train = np.sum(np.ediff1d(URM_train_final.indptr) == 0)
        user_no_item_validation = np.sum(np.ediff1d(URM_validation_final.indptr) == 0)

        folds[k] = [URM_train_final, URM_validation_final]

        print("Fold n°" + str(k) + " loaded")
        if user_no_item_train != 0:
            print("Warning: {} ({:.2f} %) of {} users have no train items".format(user_no_item_train,
                                                                                  user_no_item_train / num_users * 100,
                                                                                  num_users))
        if user_no_item_validation != 0:
            print("Warning: {} ({:.2f} %) of {} users have no sampled items".format(user_no_item_validation,
                                                                                    user_no_item_validation / num_users * 100,
                                                                                    num_users))

    return folds

import numpy as np

from RecSysCourseMaterial.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from RecSysCourseMaterial.Recommenders.Recommender_utils import check_matrix


class MINT_Ensembler(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "MINT_Ensembler"

    def __init__(self, URM_train, recommender_0, recommender_1,
                 recommender_2, recommender_3, recommender_4):
        super(MINT_Ensembler, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.rec_list = [recommender_0, recommender_1, recommender_2,
                         recommender_3, recommender_4]
        self.num_folds = len(self.rec_list)

    def fit(self, weight_0, weight_1, weight_2, weight_3, weight_4):
        self.weight_list = [weight_0, weight_1, weight_2, weight_3, weight_4]
        self.weight_list /= np.sum(self.weight_list)

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights = []
        for index in range(self.num_folds):
            elem = self.rec_list[index]._compute_item_score(user_id_array, None) * self.weight_list[index]
            elem[np.isnan(elem)] = 0
            item_weights.append(elem)
        return np.sum(item_weights, axis=0)
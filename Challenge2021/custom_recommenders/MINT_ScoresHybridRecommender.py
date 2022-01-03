from numpy import linalg as LA

from RecSysCourseMaterial.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from RecSysCourseMaterial.Recommenders.Recommender_utils import check_matrix


class MINT_ScoresHybridRecommender4(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "MINT_ScoresHybridRecommender4"

    def __init__(self, URM_train, recommender_1, recommender_2,
                 recommender_3, recommender_4):
        super(MINT_ScoresHybridRecommender4, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4


    def fit(self, weight_1, weight_2, weight_3, weight_4, normalize=True):
        self.normalize = normalize
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.weight_3 = weight_3
        self.weight_4 = weight_4
        self.weight_sum = weight_1 + weight_2 + weight_3 + weight_4

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array, None)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array, None)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array, None)
        item_weights_4 = self.recommender_4._compute_item_score(user_id_array, None)
        if self.normalize:
            item_weights_1 /= LA.norm(item_weights_1, 2)
            item_weights_2 /= LA.norm(item_weights_2, 2)
            item_weights_3 /= LA.norm(item_weights_3, 2)
            item_weights_4 /= LA.norm(item_weights_4, 2)

        item_weights = (item_weights_1 * self.weight_1 +
                        item_weights_2 * self.weight_2 +
                        item_weights_3 * self.weight_3 +
                        item_weights_4 * self.weight_4)/self.weight_sum

        return item_weights


class MINT_ScoresHybridRecommender3(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "MINT_ScoresHybridRecommender3"

    def __init__(self, URM_train, recommender_1, recommender_2,
                 recommender_3):
        super(MINT_ScoresHybridRecommender3, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3


    def fit(self, weight_1, weight_2, weight_3, normalize=True):
        self.normalize = normalize
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.weight_3 = weight_3
        self.weight_sum = weight_1 + weight_2 + weight_3

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array, None)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array, None)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array, None)
        if self.normalize:
            item_weights_1 /= LA.norm(item_weights_1, 2)
            item_weights_2 /= LA.norm(item_weights_2, 2)
            item_weights_3 /= LA.norm(item_weights_3, 2)

        item_weights = (item_weights_1 * self.weight_1 +
                        item_weights_2 * self.weight_2 +
                        item_weights_3 * self.weight_3)/self.weight_sum

        return item_weights


class MINT_ScoresHybridRecommender2(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "MINT_ScoresHybridRecommender2"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(MINT_ScoresHybridRecommender2, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2


    def fit(self, alpha, normalize=True):
        self.normalize = normalize
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array, None)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array, None)
        if self.normalize:
            item_weights_1 /= LA.norm(item_weights_1, 2)
            item_weights_2 /= LA.norm(item_weights_2, 2)

        item_weights = item_weights_1 * (1 - self.alpha) + item_weights_2 * self.alpha

        return item_weights
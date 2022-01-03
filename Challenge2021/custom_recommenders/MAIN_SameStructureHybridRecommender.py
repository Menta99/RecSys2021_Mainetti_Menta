from RecSysCourseMaterial.Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from RecSysCourseMaterial.Recommenders.Recommender_utils import check_matrix


class MAIN_SameStructureHybridRecommender(ItemKNNCustomSimilarityRecommender):
    RECOMMENDER_NAME = "MAIN_SameStructureHybridRecommender"

    def __init__(self, URM_train, W_sparse1, W_sparse2):
        super(MAIN_SameStructureHybridRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.W_sparse1 = W_sparse1
        self.W_sparse2 = W_sparse2

    def fit(self, alpha):
        self.alpha = alpha
        new_similarity = (1 - self.alpha) * self.W_sparse1 + self.alpha * self.W_sparse2
        super(MAIN_SameStructureHybridRecommender, self).fit(new_similarity)

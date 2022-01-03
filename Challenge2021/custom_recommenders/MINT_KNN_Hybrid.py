from RecSysCourseMaterial.Recommenders.BaseRecommender import BaseRecommender
from RecSysCourseMaterial.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from RecSysCourseMaterial.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender


class MINT_KNN_Hybrid(BaseRecommender):
    RECOMMENDER_NAME = "MINT_KNN_Hybrid"

    def __init__(self, URM_train):
        super(MINT_KNN_Hybrid, self).__init__(URM_train)

    def fit(self, Item_topK, Item_shrink, Item_similarity, Item_normalize, Item_feature_weighting,
            User_topK, User_shrink, User_similarity, User_normalize, User_feature_weighting, Weight):
        # Instantiate the recommenders and the weights
        self.ItemKNN = ItemKNNCFRecommender(self.URM_train)
        self.UserKNN = UserKNNCFRecommender(self.URM_train)
        self.Weight = Weight

        # Fit the recommenders
        self.ItemKNN.fit(topK=Item_topK, shrink=Item_shrink, similarity=Item_similarity,
                         normalize=Item_normalize, feature_weighting=Item_feature_weighting)

        self.UserKNN.fit(topK=User_topK, shrink=User_shrink, similarity=User_similarity,
                         normalize=User_normalize, feature_weighting=User_feature_weighting)

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.ItemKNN._compute_item_score(user_id_array)
        item_weights_2 = self.UserKNN._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.Weight + item_weights_2 * (1 - self.Weight)

        return item_weights

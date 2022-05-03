import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender


class MainRecommender:
    def __init__(self, random_state=None):
        
        self.random_state = random_state
        self.fitted = False
        
        # Список наиболее покупаемых товаров.
        self.top_items = None
        
        # User-Item матрица.
        self.user_item_matrix = None
        self.sparse_user_item = None
        
        # Словари перевода user_id и item_id к порядковым id и наоброт.
        self.id_to_itemid = None
        self.id_to_userid = None
        self.itemid_to_id = None
        self.userid_to_id = None
        
        self.model_als = None
        self.model_own = None
        
    
    def fit(self, data_train):
        
        # Формирования списка наиболее покупаемых товаров.
        self.top_items = (
            data_train
            .groupby(by='item_id')['quantity']
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        
        # Подготовка User-Item матрицы.
        self.user_item_matrix = pd.pivot_table(data_train,
                                               index='user_id',
                                               columns='item_id',
                                               values='quantity',
                                               aggfunc='sum',
                                               fill_value=0)

        # Необходимый тип матрицы для implicit.
        self.user_item_matrix = self.user_item_matrix.astype(float) 

        # Перевод в формат saprse matrix.
        self.sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()
        
        # Словари перевода user_id и item_id к порядковым id и наоброт.
        userids = self.user_item_matrix.index.values
        itemids = self.user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))
        
        # Инициализация модели ALS.
        self.model_als = AlternatingLeastSquares(factors=10,
                                                 regularization=0.1,
                                                 iterations=15,
                                                 calculate_training_loss=True,
                                                 use_gpu=False,
                                                 random_state=self.random_state)

        # Обучение модели ALS.
        self.model_als.fit(csr_matrix(self.user_item_matrix).T.tocsr(), show_progress=True)
        
        # Инициализация модели для собственных прогнозов пользователя.
        self.model_own = ItemItemRecommender(K=1)

        # Обучение модели Own_recommender.
        self.model_own.fit(csr_matrix(self.user_item_matrix).T.tocsr(), show_progress=True)
        
        self.fitted = True
    
    
    # Метод для дополнения прогноза популярными товарами.
    def add_top_items(self, res, N):
        
        i = 0
        
        while len(res) < N:
            if self.top_items[i] not in res:
                res.append(self.top_items[i])
            
            i += 1
        
        return res
    
    
    def predict_als(self, user, N=5, other_category=999999):
        
        assert self.fitted, 'MainRecommender must be fitted before applying!'
        
        res = [self.id_to_itemid[rec[0]] for rec in self.model_als.recommend(userid=self.userid_to_id[user],
                                                                             user_items=self.sparse_user_item,
                                                                             N=N,
                                                                             filter_already_liked_items=False,
                                                                             filter_items=[self.itemid_to_id[other_category]],
                                                                             recalculate_user=True)]
        
        # Удаление дубликатов.
        res = [*set(res)]
        
        # Дополнение прогноза популярными товарами в случае недостатка товаров в прогнозе.
        res = self.add_top_items(res, N)
        
        return res
    
    
    def predict_sur(self, user, N=5, other_category=999999):
        
        assert self.fitted, 'MainRecommender must be fitted before applying!'

        # Формирование списка N похожих пользователей, кроме первого пользователя - это сам рассматриваемый пользователь.
        list_similar_users = [user_id for user_id, _ in self.model_als.similar_users(self.userid_to_id[user], N + 1)[1:]]

        # Список для хранения товаров похожих пользователей.
        res = []

        # Для каждого похожего пользователя:
        for similar_user in list_similar_users:
            # выполнить прогноз N продуктов,
            recs = self.model_own.recommend(userid=similar_user,
                                            user_items=self.sparse_user_item,
                                            N=N,
                                            filter_items=[self.itemid_to_id[other_category]])

            # выбрать первый продукт.
            item = recs[0][0]

            # Для всех предсказанных продуктов текущего похожего пользователя:
            for i in range(len(recs)):
                # если продукт уже в списке предсказанных:
                if item in res:
                    # выбрать следующий продукт.
                    item = recs[i][0]

            # добавить продукт в список для рекомендации.
            res.append(self.id_to_itemid[item])
        
        # Удаление дубликатов.
        res = [*set(res)]
        
        # Дополнение прогноза популярными товарами в случае недостатка товаров в прогнозе.
        res = self.add_top_items(res, N)
        
        return res

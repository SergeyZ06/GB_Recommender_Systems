# Функция для подбора похожих продуктов.
def get_similar_items_recommendation(user, result_train, model, id_to_itemid, itemid_to_id, filter_items=[], N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    
    # На вход функции поступает список купленных продуктов, из него выбираются топ-N купленных продуктов.
    user_items = [*result_train.loc[result_train['user_id'] == 1, 'actual']][0][:N]
    
    # Список для хранения похожих продуктов.
    res = []
    
    # Для каждого продукта из топа:
    for item in user_items:
        # выбрать второй наиболее похожий продукт, так как первый - это сам рассматриваемый продукт.
        item_rec = id_to_itemid[model.similar_items(itemid_to_id[item], N=2)[1][0]]
        
        # Если продукт попал в категорию "прочие",
        if item_rec in filter_items:
            # выбрать следующий продукт.
            item_rec = id_to_itemid[model.similar_items(itemid_to_id[item], N=3)[2][0]]
        
        res.append(id_to_itemid[model.similar_items(itemid_to_id[item], N=2)[1][0]])
    
    return res


# Функция для подбора продуктов похожих пользователей.
def get_similar_users_recommendation(user, model_als, model_own, user_item_matrix, filter_items=[], N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    # Формирование списка N похожих пользователей, кроме первого пользователя - это сам рассматриваемый пользователь.
    list_similar_users = [user_id for user_id, _ in model_als.similar_users(user, N + 1)[1:]]
    
    # Список для хранения товаров похожих пользователей.
    res = []
    
    # Для каждого похожего пользователя:
    for similar_user in list_similar_users:
        # выполнить прогноз N продуктов,
        recs = model_own.recommend(userid=similar_user,
                                   user_items=user_item_matrix,
                                   N=N,
                                   filter_items=filter_items)
        
        # выбрать первый продукт.
        item = recs[0][0]
        
        # Для всех предсказанных продуктов текущего похожего пользователя:
        for i in range(len(recs)):
            # если продукт уже в списке предсказанных:
            if item in res:
                # выбрать следующий продукт.
                item = recs[i][0]

        # добавить продукт в список для рекомендации.
        res.append(item)
    
    return res

# Функция для фильтрации продуктов.
def prefilter_items(retail, products=None, departments=[], items_too_cheap=[], items_too_expensive=[]):
    # Уберем самые популярные товары (их и так купят).
    popularity = retail.groupby('item_id')['user_id'].nunique().reset_index()
    popularity = popularity.rename(columns={'user_id': 'share_unique_users'})
    popularity['share_unique_users'] = popularity['share_unique_users'] / retail['user_id'].nunique()
    top_popular = popularity.query('share_unique_users > 0.5')['item_id']
    retail = retail[~retail['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят).
    top_not_popular = popularity.query('share_unique_users < 0.01')['item_id']
    retail = retail[~retail['item_id'].isin(top_not_popular)]

    # Уберем товары, которые не продавались за последние 12 месяцев.
    weeks_actual = retail['week_no'].unique()[-12:]
    items_actual = retail.loc[retail['week_no'].isin(weeks_actual), 'item_id'].unique()
    retail = retail[retail['item_id'].isin(items_actual)]

    # Уберем не интересные для рекоммендаций категории (department).
    if products is not None and departments != []:
        items_dep = products.loc[products['department'].isin(departments), 'item_id']
        retail = retail[retail['item_id'].isin(items_dep)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    retail = retail[~retail['item_id'].isin(items_too_cheap)]

    # Уберем слишком дорогие товары.
    retail = retail[~retail['item_id'].isin(items_too_expensive)]

    return retail


def postfilter_items(user_id, recommednations):
    pass

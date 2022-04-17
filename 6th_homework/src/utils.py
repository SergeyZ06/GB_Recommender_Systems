import numpy as np
import pandas as pd


def prefilter_items(data: pd.DataFrame,
                    feature_item_id: str,
                    feature_to_top: str,
                    top: int = 5000,
                    other_category: int = 999999) -> pd.DataFrame:
    
    '''
    Function for getting only top items' ids.
    Other items' ids will be replaced with 'other_category' value.
    
    data : pd.DataFrame to filter.
    
    feature_item_id : str contains feature name "item id" for grouping,
    
    feature_to_top : str contains feature name for sorting top items.
    
    top : int, how many items will be taken.
    
    other_category : int, new items id outside the top.
    '''
    
    top_items = (
        data
        .groupby(by=feature_item_id)[feature_to_top]
        .sum()
        .reset_index()
        .sort_values(by=feature_to_top, ascending=False)
        .head(top)
    )
    
    data_filtered = data.copy()
    data_filtered.loc[~data_filtered[feature_item_id].isin(top_items[feature_item_id]), feature_item_id] = other_category
    
    return data_filtered


def train_test_split(data: pd.DataFrame,
                     feature_to_split: str,
                     split_value: 'int | float') -> 'pd.DataFrame & pd.DataFrame':
    
    '''
    Function for deviding orifinal dataset into two subsets: train and test.
    
    data : pd.DataFrame, original dataset.
    
    feature_to_split : str, which feature original dataset will be splitted by.
    
    split_value : int or float, threshold for splitting,
    '''
    
    data_train = data[data[feature_to_split] < split_value].copy()
    data_test = data[data[feature_to_split] >= split_value].copy()
    
    return data_train, data_test


def prepare_result(data: pd.DataFrame,
                   user_id: str = 'user_id',
                   item_id: str = 'item_id') -> pd.DataFrame:
    
    '''
    Function for preparing result dataset with users' and items' ids.
    
    data : pd.DataFrame, dataset with history of purchases.
    
    user_id : str, containts feature name of users' ids.
    
    item_id : str, containts feature name of items' ids.
    '''
    
    result = data.copy().groupby('user_id')['item_id'].unique().reset_index()
    result.columns=['user_id', 'actual']
    return result

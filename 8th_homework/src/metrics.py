import numpy as np
import pandas as pd


def precision_at_k(series_actual: pd.Series,
                   series_predicted: pd.Series,
                   K: int = 5,
                   return_series: bool = False) -> float:
    '''
    This function is to calculate metric 'Precision@K' for recommender system.
    
    Parameters
    ----------
    series_actual : pandas Series with list of actual items' ids.
    
    series_predicted : pandas Series with list of predicted items' ids.
    
    K = 5 : int, number of first K recommended items will be considered,
        if None - all recommended items will be considered.
                            
    return_series = False : bool, if False, common float type metric will be returned.
        If True, np.Series type metric will be returned for each row.
    
    
    Examples
    --------
    Constructing DataFrame from a dictionary.
    
    >>> df = pd.DataFrame([{'user_id': 99, 'actual_items': [1, 2, 3, 4, 5, 6, 7], 'predicted_items': [2, 3, 4, 5, 6, 7, 8]},
    >>>                    {'user_id': 52, 'actual_items': [8, 9, 10, 11, 12, 13, 14], 'predicted_items': [2, 3, 4]}])
    
    Function returns float type metric 'Precision@K'.
    
    >>> precision_at_k(df['actual_items'], df['predicted_items'])
    0.5
    
    Function returns np.Series type metric 'Precision@K' for each row.
    
    >>> precision_at_k(df['actual_items'], df['predicted_items'], K=None, return_series=True)
    0    1.0
    1    0.0
    Name: precision, dtype: float64
    '''
    
    if type(series_actual) != pd.Series:
        raise Exception('Parametr "series_actual" must be pandas.Series type!')
    
    if type(series_predicted) != pd.Series:
        raise Exception('Parametr "series_predicted" must be pandas.Series type!')
    
    if type(K) != int and K is not None:
        raise Exception('Parametr "K" must be int type or None!')
    
    if type(return_series) != bool:
        raise Exception('Parametr "return_series" must be bool type!')
    
    result = pd.concat(objs=[series_actual, series_predicted], axis=1)
    result.columns = ['actual', 'predicted']
    result['precision'] = result.apply(lambda row: np.mean(np.isin(row['predicted'][:K], row['actual'])), axis=1)
    
    if return_series:
        return result['precision']
    
    return result['precision'].mean()


def recall_at_k(series_actual: pd.Series,
                series_predicted: pd.Series,
                K: int = 5,
                return_series: bool = False) -> float:
    '''
    This function is to calculate metric 'Recall@K' for recommender system.
    
    Parameters
    ----------
    series_actual : pandas Series with list of actual items' ids.
    
    series_predicted : pandas Series with list of predicted items' ids.
    
    K = 5 : int, number of first K recommended items will be considered,
        if None - all recommended items will be considered.
                            
    return_series = False : bool, if False, common float type metric will be returned.
        If True, np.Series type metric will be returned for each row.
    
    
    Examples
    --------
    Constructing DataFrame from a dictionary.
    
    >>> df = pd.DataFrame([{'user_id': 99, 'actual_items': [1, 2, 3, 4, 5, 6, 7], 'predicted_items': [2, 3, 4, 5, 6, 7, 8]},
    >>>                    {'user_id': 52, 'actual_items': [8, 9, 10, 11, 12, 13, 14], 'predicted_items': [2, 3, 4]}])
    
    Function returns float type metric 'Recall@K'.
    
    >>> recall_at_k(df['actual_items'], df['predicted_items'])
    0.35714285714285715
    
    Function returns np.Series type metric 'Recall@K' for each row.
    
    >>> recall_at_k(df['actual_items'], df['predicted_items'], K=None, return_series=True)
    0    0.857143
    1    0.000000
    Name: recall, dtype: float64
    '''
    
    if type(series_actual) != pd.Series:
        raise Exception('Parametr "series_actual" must be pandas.Series type!')
    
    if type(series_predicted) != pd.Series:
        raise Exception('Parametr "series_predicted" must be pandas.Series type!')
    
    if type(K) != int and K is not None:
        raise Exception('Parametr "K" must be int type or None!')
    
    if type(return_series) != bool:
        raise Exception('Parametr "return_series" must be bool type!')
    
    result = pd.concat(objs=[series_actual, series_predicted], axis=1)
    result.columns = ['actual', 'predicted']
    result['recall'] = result.apply(lambda row: np.mean(np.isin(row['actual'], row['predicted'][:K])), axis=1)
    
    if return_series:
        return result['recall']
    
    return result['recall'].mean()

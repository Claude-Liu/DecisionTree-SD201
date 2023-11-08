from typing import List
import numpy as np

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    expected_results = np.array(expected_results)
    actual_results = np.array(actual_results)
    num_pred_pos = (actual_results==True).sum()
    num_tp = ((actual_results==True) & (expected_results==True)).sum()
    precision = num_tp/num_pred_pos

    num_label_pos = (expected_results==True).sum()
    recall = num_tp/num_label_pos
    return precision, recall

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    precision, recall = precision_recall(expected_results, actual_results)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score

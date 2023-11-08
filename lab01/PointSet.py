from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        num_pos = (self.labels==1.0).sum()
        num_neg = (self.labels==0.0).sum()
        num = len(self.labels)
        gini = 1 - (num_pos/num)**2 - (num_neg/num)**2
        return gini


    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        all_gini = []
        # get the transpose of the features matrix
        features_transpose = self.features.T
        gini = self.get_gini()
        for feature in features_transpose:
            # get the slice of self.labels for which the feature is 0
            labels_0 = self.labels[feature==0.0]
            features_0 = self.features[feature==0.0]
            pointSet_0 = PointSet(features_0, labels_0, self.types)
            # get the slice of self.labels for which the feature is 1
            labels_1 = self.labels[feature==1.0]
            features_1 = self.features[feature==1.0]
            pointSet_1 = PointSet(features_1, labels_1, self.types)
            gini_gain = gini-((len(labels_0)/len(self.labels))*pointSet_0.get_gini() + (len(labels_1)/len(self.labels))*pointSet_1.get_gini())
            all_gini +=[gini_gain]
        best_feature = np.argmax(np.array(all_gini))
        best_gini = np.array(all_gini).max()
        #print(best_feature)
        if len(self.labels[features_transpose[best_feature]==0.0])==0 or len(self.labels[features_transpose[best_feature]==1.0])==0:
            return None, None
        return best_feature, best_gini


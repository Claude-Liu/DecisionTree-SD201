from typing import List

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        
        self.points = PointSet(features, labels, types)
        features_transpose = self.points.features.T
        #print(self.points.labels)
        #print(features_transpose)
        self.best_feature, _ = self.points.get_best_gain()

        #print(self.best_feature)
        if self.best_feature == None:
            self.chilfPoints_0 = None
            self.childPoints_1 = None
            return
        labels_0 = self.points.labels[features_transpose[self.best_feature]==0.0]
        features_0 = self.points.features[features_transpose[self.best_feature]==0.0]
        self.childPoints_0 = PointSet(features_0, labels_0, self.points.types)

        labels_1 = self.points.labels[features_transpose[self.best_feature]==1.0]
        features_1 = self.points.features[features_transpose[self.best_feature]==1.0]
        self.childPoints_1 = PointSet(features_1, labels_1, self.points.types)
        

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        def vote(labels):
            return (labels==True).sum() > (labels==False).sum()
        
        if self.best_feature == None:
            return vote(self.points.labels)
        critical_feature = features[self.best_feature]
        if critical_feature == 0.0:
            return vote(self.childPoints_0.labels)
        else:
            return vote(self.childPoints_1.labels)


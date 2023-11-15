from typing import List

from PointSet import PointSet, FeaturesTypes
from Node import Node

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
                 h: int = 1,
                 min_split_points: int = 1):
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
        
        self.root = Node(PointSet(features, labels, types))
        self.split_all(self.root, h, min_split_points)
        print("Tree is built successfully!")

    def split_all(self,current_node: Node, h: int, min_split_points: int):
        """split a give node to a subtree in decisionb tree by recursion"""
        if h==0:
            return
        if len(current_node) <= min_split_points:
            return
        if len(current_node) < 2*min_split_points:
            return
        if current_node.split(min_split_points)==True:
            self.split_all(current_node.leftChild, h-1, min_split_points)
            self.split_all(current_node.rightChild, h-1, min_split_points)
        else:
            return


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
        return self.decide_(features, self.root)

    def decide_(self, features, current_node: Node):
        """predict the label of a point using the tree by recursion"""
        def vote(labels):
            return (labels==True).sum() > (labels==False).sum()
        
        if current_node.best_feature == None:
            return vote(current_node.pointSet.labels)
        critical_feature = features[current_node.best_feature]
        type = current_node.pointSet.types[current_node.best_feature]
        if type == FeaturesTypes.BOOLEAN:
            if critical_feature == 0:
                return self.decide_(features, current_node.leftChild)
            else:
                return self.decide_(features, current_node.rightChild)
        elif type == FeaturesTypes.CLASSES:
            if critical_feature == current_node.threshold:
                return self.decide_(features, current_node.leftChild)
            else:
                return self.decide_(features, current_node.rightChild)
        elif type == FeaturesTypes.REAL:
            if critical_feature < current_node.threshold:
                return self.decide_(features, current_node.leftChild)
            else:
                return self.decide_(features, current_node.rightChild)
        else:
            raise(Exception('Unknown feature type'))

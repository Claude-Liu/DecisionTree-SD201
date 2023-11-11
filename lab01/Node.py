from typing import List

from PointSet import PointSet, FeaturesTypes

class Node:

    def __init__(self, pointSet: PointSet):
        self.pointSet = pointSet
        self.best_feature = None
        self.threshold = None
        self.leftChild = None
        self.rightChild = None

    def split(self, min_split_points: int):
        leftPoint, rightPoint, best_feature, threshold = self.pointSet.get_best_split(min_split_points)
        if leftPoint is None and rightPoint is None:
            return False
        if len(leftPoint.labels) < min_split_points or len(rightPoint.labels) < min_split_points:
            return False
        else:
            self.leftChild = Node(leftPoint)
            self.rightChild = Node(rightPoint)
            self.best_feature = best_feature
            self.threshold = threshold
            return True
    
    def __len__(self):
        return len(self.pointSet.labels)
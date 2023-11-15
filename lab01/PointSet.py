from typing import List, Tuple

from enum import Enum
import numpy as np

MIN_LIMIT = -1

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
        self.best_feature = None
        self.best_gini = None
        self.best_threshold = None
        self.best_split_found = False

    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        if len(self.labels)==0:
            return 0.0
        num_pos = (self.labels==1.0).sum()
        num_neg = (self.labels==0.0).sum()
        num = len(self.labels)
        gini = 1 - (num_pos/num)**2 - (num_neg/num)**2
        return gini
    
    def get_gini_gain(self,feature,type,gini, min_split_points):
        if type==FeaturesTypes.BOOLEAN:
            pointSet_0, pointSet_1 = self.split_point_set(feature, 0.0, type)
            if 0<len(pointSet_0.labels) < min_split_points or 0<len(pointSet_1.labels) < min_split_points:
                gini_gain = MIN_LIMIT
            else:
                gini_gain = gini-((len(pointSet_0.labels)/len(self.labels))*pointSet_0.get_gini() + (len(pointSet_1.labels)/len(self.labels))*pointSet_1.get_gini())
            threshold = None
        elif type==FeaturesTypes.CLASSES:
            # set initial value to -1 so that we will always find a better value (0)
            gini_gain = MIN_LIMIT
            threshold = None
            unique_feature = np.unique(feature)
            for i in unique_feature:
                pointSet_0, pointSet_1 = self.split_point_set(feature, i, type)
                if 0<len(pointSet_0.labels) < min_split_points or 0<len(pointSet_1.labels) < min_split_points:
                    continue
                gini_gain_i = gini-((len(pointSet_0.labels)/len(self.labels))*pointSet_0.get_gini() + (len(pointSet_1.labels)/len(self.labels))*pointSet_1.get_gini())
                if gini_gain_i > gini_gain:
                    gini_gain = gini_gain_i
                    threshold = i
            # assert threshold != None
        elif type==FeaturesTypes.REAL:
            # sort the feature with complexity O(nlogn)
            sorted_indices = np.argsort(feature)
            sorted_feature = feature[sorted_indices]
            sorted_labels = self.labels[sorted_indices]
            threshold = sorted_feature[0]
            gini_gain = 0
            last_threshold = sorted_feature[0]
            last_threshold_index = 0
            num_feature_0_label_0 = 0
            num_feature_0_label_1 = 0
            num_feature_1_label_0 = (self.labels==0.0).sum()
            num_feature_1_label_1 = (self.labels==1.0).sum()
            for i, f in enumerate(sorted_feature):
                if i==0:
                    continue
                if f==last_threshold:
                    continue
                for j in range(last_threshold_index, i):
                    if sorted_labels[j]==0.0:
                        num_feature_0_label_0 += 1
                        num_feature_1_label_0 -= 1
                    else:
                        num_feature_0_label_1 += 1
                        num_feature_1_label_1 -= 1
                num_feature_0 = num_feature_0_label_0 + num_feature_0_label_1
                num_feature_1 = num_feature_1_label_0 + num_feature_1_label_1
                
                if 0< num_feature_0 < min_split_points or 0< num_feature_1 < min_split_points:
                    last_threshold = f
                    last_threshold_index = i
                    continue
                gini_0 = 1 - (num_feature_0_label_0/num_feature_0)**2 - (num_feature_0_label_1/num_feature_0)**2
                gini_1 = 1 - (num_feature_1_label_0/num_feature_1)**2 - (num_feature_1_label_1/num_feature_1)**2
                gini_gain_i = gini-((num_feature_0/len(self.labels))*gini_0 + (num_feature_1/len(self.labels))*gini_1)
                if gini_gain_i > gini_gain:
                    gini_gain = gini_gain_i
                    threshold = (f+last_threshold)/2
                last_threshold = f
                last_threshold_index = i

            '''
            # sort the feature with complexity O(n^2)
            feature_sorted = np.sort(np.unique(feature))
            for i,thres in enumerate(feature_sorted):
                pointSet_0, pointSet_1 = self.split_point_set(feature, thres, type)
                if 0<len(pointSet_0.labels) < min_split_points or 0<len(pointSet_1.labels) < min_split_points:
                    continue
                gini_gain_i = gini-((len(pointSet_0.labels)/len(self.labels))*pointSet_0.get_gini() + (len(pointSet_1.labels)/len(self.labels))*pointSet_1.get_gini())
                if gini_gain_i > gini_gain:
                    gini_gain = gini_gain_i
                    if i-1>=0:
                        # threshold is the mean of the two closest values from the two splits
                        thres = (feature_sorted[i-1]+thres)/2

                    threshold = thres
            assert threshold != None
            '''
        else:
            raise(Exception("Unknown type"))
            
        return gini_gain, threshold

    def get_best_gain_threshold(self, min_split_points=1) -> Tuple[int, float]:
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
        all_threshold = []
        # get the transpose of the features matrix
        features_transpose = self.features.T
        if len(self.labels)==0:
            return None, None, None
        gini = self.get_gini()
        for feature, type in zip(features_transpose, self.types):
            gini_gain, threshold = self.get_gini_gain(feature, type, gini, min_split_points)
            all_gini +=[gini_gain]
            all_threshold += [threshold]
        best_feature = np.argmax(np.array(all_gini))
        best_shreshold = all_threshold[best_feature]
        best_gini = np.array(all_gini).max()
        if best_gini==MIN_LIMIT:
            return None, None, None
        # check if the split is possible
        if self.types[best_feature] == FeaturesTypes.BOOLEAN:
            if len(self.labels[features_transpose[best_feature]==0.0])==0 or len(self.labels[features_transpose[best_feature]==1.0])==0:
                return None, None, None
        elif self.types[best_feature] == FeaturesTypes.CLASSES:
            if len(self.labels[features_transpose[best_feature]==best_shreshold])==0 or len(self.labels[features_transpose[best_feature]!=best_shreshold])==0:
                return None, None, None
        elif self.types[best_feature] == FeaturesTypes.REAL:
            if len(self.labels[features_transpose[best_feature]<best_shreshold])==0 or len(self.labels[features_transpose[best_feature]>=best_shreshold])==0:
                return None, None, None 
        else:
            raise(Exception("Unknown type"))
        return best_feature, best_gini, best_shreshold

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
        best_feature, best_gini, best_threshold = self.get_best_gain_threshold()
        self.best_split_found = True
        self.best_feature = best_feature
        self.best_gini = best_gini
        self.best_threshold = best_threshold
        return best_feature, best_gini
    
    def get_best_threshold(self) -> float:
        '''
        compute the best threshold for the best feature
        '''
        if self.best_split_found == False:
            raise(Exception("No split has been found yet"))
        # if we need we distinguish the reasons we get None: no split possible or Boolean feature
        return self.best_threshold
    
    
    def split_point_set(self, feature: List[float], threshold: int, type: FeaturesTypes) -> Tuple['PointSet', 'PointSet']:
        """Split the set of points along a feature"""
        
        if type==FeaturesTypes.BOOLEAN:
            labels_0 = self.labels[feature==0.0]
            features_0 = self.features[feature==0.0]
            pointSet_0 = PointSet(features_0, labels_0, self.types)
            labels_1 = self.labels[feature==1.0]
            features_1 = self.features[feature==1.0]
            pointSet_1 = PointSet(features_1, labels_1, self.types)
        elif type==FeaturesTypes.CLASSES:
            labels_0 = self.labels[feature==threshold]
            features_0 = self.features[feature==threshold]
            pointSet_0 = PointSet(features_0, labels_0, self.types)
            labels_1 = self.labels[feature!=threshold]
            features_1 = self.features[feature!=threshold]
            pointSet_1 = PointSet(features_1, labels_1, self.types)
        elif type==FeaturesTypes.REAL:
            labels_0 = self.labels[feature<threshold]
            features_0 = self.features[feature<threshold]
            pointSet_0 = PointSet(features_0, labels_0, self.types)
            labels_1 = self.labels[feature>=threshold]
            features_1 = self.features[feature>=threshold]
            pointSet_1 = PointSet(features_1, labels_1, self.types)
        else:
            raise(Exception("Unknown type"))
        return pointSet_0, pointSet_1
    
    def get_best_split(self, min_split_points) -> Tuple[int, float, 'PointSet', 'PointSet']:
        """Compute the best split of the set of points"""
        best_feature_id, _, threshold = self.get_best_gain_threshold(min_split_points)
        if best_feature_id == None:
            return None, None, best_feature_id, threshold
        features_transpose = self.features.T
        best_feature = features_transpose[best_feature_id]
        pointSet_0, pointSet_1 = self.split_point_set(best_feature, threshold=threshold, type=self.types[best_feature_id])
        return pointSet_0, pointSet_1, best_feature_id, threshold

    
    
    


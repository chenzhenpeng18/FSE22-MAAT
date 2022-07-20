import sys
sys.path.append("../")
from collections import defaultdict
import random
import numpy as np
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from shapely.geometry import Polygon, Point, LineString
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import math
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef


def get_classifier(name):
    """ Creates a default classifier based on name.

    Parameters:
        name (str) -- Name of the classifier
    Returns:
        clf (classifer) -- Classifier with default configuration from scipy
    """
    if name == "lr":
        clf = LogisticRegression()
    elif name == "dt":
        clf = tree.DecisionTreeClassifier()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "bayes":
        clf = GaussianNB()
    if name == "rf":
        clf = RandomForestClassifier()
    return clf


def create_baseline(clf_name,dataset_orig, privileged_groups,unprivileged_groups,
                    data_splits=50,repetitions=50,odds={"0":[1,0],"1":[0,1]},options = [0,1],
                   degrees = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],verbose=False):
    """ Create a baseline by mutating predictions of an original classification model (clf_name).

    Parameters:
        clf_name (str)          -- Name of the original classifier to mutate
        dataset_orig (dataset)  -- Dataset used for training and testing
        privileged_groups (list) -- Attribute and label of privileged group
        unprivileged_groups(list)--Attribute and label of unprivileged group
        data_splits (int)       -- Number of different datasplits 
        repetitions (int)       -- Number of repetitions of mutation process for each datasplit
        odds (dict)             -- Odds for mutation. Keys determine the "name" of mutation strategy, values the odds for each label 
        options (list)          -- Available labels to mutate predictions
        degrees (list)          -- Mutation degrees that are used to create baselines
        verbose (bool)          -- Outputs number of current datasplit
        
    Returns:
        results (dict) -- dictionary of mutation results (one entry for each key in odds)
            dictonary values are list (mutation degree) of lists (performance for each datasplit X repetitions)
    """
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    ids = [x for x in range(len(dataset_orig_test.labels))]
    l = len(dataset_orig_test.labels)

    results = defaultdict(lambda: defaultdict(list))
    
    
    # Iterate over different datasplits
    for s in range(data_splits):
        if verbose:
            print ("Current datasplit:",s)
        np.random.seed(s)
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        scaler = MinMaxScaler()
        dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
        dataset_orig_test.features = scaler.transform(dataset_orig_test.features)
        
        # Make initial predictions
        clf = get_classifier(clf_name)
        clf = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)
        pred = clf.predict(dataset_orig_test.features).reshape(-1,1)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.labels = pred
        
        # Mutate labels for each degree
        for degree in degrees:
            # total number of labels to mutate
            to_mutate = int(l*degree)

            for name,o in odds.items():
                # Store each mutation attempt
                hist = []
                for _ in range(repetitions):
                    # Generate new random labels
                    rand = np.random.choice(options, to_mutate, p=o)
                    # Select prediction ids that are being mutated
                    to_change = np.random.choice(ids, size=to_mutate, replace=False)
                    changed = np.copy(pred)
                    for t,r in zip(to_change, rand):
                        changed[t] = r
                    
                    # Determine accuray and fairness of mutated model 
                    dataset_orig_test_pred.labels = changed
                    class_metric = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                     unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                    spd = abs(class_metric.statistical_parity_difference())
                    aaod = abs(class_metric.average_abs_odds_difference())
                    eod = abs(class_metric.equal_opportunity_difference())
                    acc = class_metric.accuracy()
                    macrop = precision_score(dataset_orig_test.labels, dataset_orig_test_pred.labels, average='macro')
                    macror = recall_score(dataset_orig_test.labels, dataset_orig_test_pred.labels, average='macro')
                    macrof1=f1_score(dataset_orig_test.labels, dataset_orig_test_pred.labels, average='macro')
                    mcc=matthews_corrcoef(dataset_orig_test.labels, dataset_orig_test_pred.labels)
                    hist.append([spd, aaod, eod, acc, macrop, macror, macrof1, mcc])
                results[name][degree] += hist
    return results

def normalize(base_accuray,base_fairness,method_dict=dict()):
    """ Normalize baseline and bias mitigation methods within the range of the baseline.

    Parameters:
        base_accuray (list)  -- Accuracy at each mutation degree
        base_fairness (list) -- Fairness at each mutation degree
        method_dict (dict)   -- Accuracy and fairness of bias mitigation methods
        
    Returns:
        normalized_accuracy (list) -- Normalized accuracy at each mutation degree
        normalized_accuracy (list) -- Normalized fairness at each mutation degree
        normalized_methods (list) -- Normalized accuracy and fairness of bias mitigation methods
    """

    # Determine range of values 
    range_accuracy = np.max(base_accuray)-np.min(base_accuray)
    range_fairness = np.max(base_fairness)-np.min(base_fairness)
    min_accuracy = np.min(base_accuray)
    min_fairness = np.min(base_fairness)
    # Normalize values
    normalized_fairness = (base_fairness-min_fairness)/range_fairness
    normalized_accuracy = (base_accuray-min_accuracy)/range_accuracy
    
    # Normalize values of bias mitigation methods
    normalized_methods = dict()
    for k, (acc,fair) in method_dict.items():
        norm_acc = (acc-min_accuracy)/range_accuracy
        norm_fair = (fair-min_fairness)/range_fairness
        normalized_methods[k] = (norm_acc,norm_fair)
    return normalized_accuracy, normalized_fairness, normalized_methods



def classify_region(base, normalized_methods):
    """ Determine bias mitigation region of normalized bias mitigation methods.

    Parameters:
        base (LineString)  -- Geometrical line (LineString) of normalized baseline created with shapely
        normalized_methods (dict) -- Normalized accuracy and fairness of bias mitigation methods
        
    Returns:
        mitigation_regions (dict) -- Bias mitigation region for each normalized bias mitigation method
    """
    mitigation_regions = dict()
    for k,(acc,fair) in normalized_methods.items():
        # define a point for each bias mitigation method
        p = Point(fair,acc)
        # Extend bias mitigation point towards four directions (left,right,up,down)
        line_down = LineString([(p.x, p.y),(p.x, 0)])
        line_right = LineString([(p.x, p.y),(2, p.y)])
        line_up = LineString([(p.x, p.y),(p.x, 2)])
        line_left = LineString([(p.x, p.y),(0, p.y)])
        # Determine bias mitigation region based on intersection with baseline
        if base.intersects(line_down) and base.intersects(line_right):
            mitigation_regions[k] = "good"
        elif base.intersects(line_down):
            mitigation_regions[k] = "win-win"
        elif base.intersects(line_up):
            mitigation_regions[k] = "bad"
        elif base.intersects(line_left):
            mitigation_regions[k] = "lose-lose"
        elif fair < 0:
            mitigation_regions[k] = "lose-lose"
        else:
            mitigation_regions[k] = "inverted"
    return mitigation_regions

def cut(line, distance):
    """ Cuts a line in two parts, at a distance from its starting point

    Parameters:
        line (LineString)  -- Geometrical line (LineString) of to be cut, created with shapely
        distance (float) -- Distance from origin (first point) of line where the cut should be place
        
    Returns:
        LineString,LineString -- Left and right part of original line, cut at the specified distance
    """
    # Check whether line cut is possible
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)

    # Iterate each point of line (Line = (point1, point2 ...)) to find position of cut
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

def compute_area(base,method):
    """ Compute area a bias mitigation method forms with the baseline, 
        by connection them with a horizontal and vertical line.

    Parameters:
        base (LineString)  -- Geometrical line (LineString) of normalized baseline created with shapely
        method (tuple)     -- Normalized accuracy and fairness of a bias mitigation method
        
    Returns:
        area (float) -- Bias mitigation region for each normalized bias mitigation method
    """

    # Create Point for bias mitigation method performance
    acc,fair = method
    p = Point(fair,acc)

    # Create horizontal and vertical line to connect point with baseline
    line_down = LineString([(p.x, p.y),(p.x, 0)])
    line_right = LineString([(p.x, p.y),(1, p.y)])

    # find intersection
    down_inter = base.intersection(line_down)
    right_inter = base.intersection(line_right)
    
    # Create a Polygon with the bias mitigation method point, and the intersections with the baseline
    cut_right,cut_left=cut(base,base.project(down_inter))
    cut_right,cut_left=cut(cut_right,base.project(right_inter))
    area = [(p.x,p.y)] + list(cut_left.coords) + [(p.x,p.y)]
    poly = Polygon(area)
    return poly.area
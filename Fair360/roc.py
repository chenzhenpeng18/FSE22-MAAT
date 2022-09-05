import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import mean, std
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score
import pandas as pd
from utility import get_data,get_classifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import os
import argparse

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'german', 'compas', 'bank', 'mep'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected attribute")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
attr = args.protected
clf_name = args.clf

metric_name = "Statistical parity difference"

val_name = "roc_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used, attr)

metric_ub = 0.05
metric_lb = -0.05

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd',  'aod', 'eod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 50
for r in range(repeat_time):
    print (r)
    np.random.seed(r)

    # split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=[attr])
    
    clf = get_classifier(clf_name)

    if clf_name == 'svm':
        clf = CalibratedClassifierCV(base_estimator = clf)

    clf = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)
    
    
    pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]
    train_pred = clf.predict(dataset_orig_train.features).reshape(-1,1)
    train_prob = clf.predict_proba(dataset_orig_train.features)[:,pos_ind].reshape(-1,1)

    pred = clf.predict(dataset_orig_test.features).reshape(-1,1)
    pred_prob = clf.predict_proba(dataset_orig_test.features)[:,pos_ind].reshape(-1,1)
    
    
    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = train_pred
    dataset_orig_train_pred.scores = train_prob


    dataset_orig_test_pred = dataset_orig_test.copy()
    dataset_orig_test_pred.labels = pred
    dataset_orig_test_pred.scores = pred_prob
    
    
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name=metric_name,
                                  metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_train, dataset_orig_train_pred)
    pred_roc = ROC.predict(dataset_orig_test_pred)

    round_result = measure_final_score(dataset_orig_test, pred_roc, privileged_groups, unprivileged_groups)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index + '\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\t%f\n' % (mean(results[p_index]), std(results[p_index])))
fout.close()
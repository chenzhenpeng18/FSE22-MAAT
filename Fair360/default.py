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
import os
import argparse
import copy

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

val_name = "default_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used, attr)

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd',  'aod', 'eod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 50
for r in range(repeat_time):
    print (r)

    np.random.seed(r)
    #split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                             protected_attribute_names=[attr])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])

    clf = get_classifier(clf_name)
    clf = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)

    test_df_copy = copy.deepcopy(dataset_orig_test)
    pred_de = clf.predict(dataset_orig_test.features)
    test_df_copy.labels = pred_de

    round_result= measure_final_score(dataset_orig_test,test_df_copy,privileged_groups,unprivileged_groups)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index+'\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\t%f\n' % (mean(results[p_index]),std(results[p_index])))
fout.close()

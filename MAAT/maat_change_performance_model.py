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
from utility import get_data
from sklearn.model_selection import train_test_split
import os
import argparse
import copy
from WAE import data_dis
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from time import *

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "dt":
        clf = DecisionTreeClassifier()
    elif name == "svm":
        clf = CalibratedClassifierCV(base_estimator = LinearSVC())
    elif name == "bayes":
        clf = BernoulliNB()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'german', 'compas', 'bank', 'mep'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'bayes', 'dt','lr'], help="Classifier name")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected attribute")

args = parser.parse_args()

dataset_used = args.dataset
attr = args.protected
algo = args.clf

val_name = "./maat_lrfairmodel_{}_{}_{}.txt".format(algo,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used, attr)

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd',  'aod', 'eod']
for p_index in performance_index:
    results[p_index] = []

pp_result = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd',  'aod', 'eod']
for p_index in performance_index:
    pp_result[p_index] = []


repeat_time = 50

for r in range(repeat_time):
    print (r)

    np.random.seed(r)
    #split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    dataset_orig_train_new = data_dis(pd.DataFrame(dataset_orig_train),attr,dataset_used)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train_new)
    dataset_orig_train_new = pd.DataFrame(scaler.transform(dataset_orig_train_new), columns=dataset_orig.columns)
    dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                             protected_attribute_names=[attr])
    dataset_orig_train_new = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                             label_names=['Probability'], protected_attribute_names=[attr])

    clf_p = get_classifier(algo)
    clf_p = clf_p.fit(dataset_orig_train.features, dataset_orig_train.labels)

    pred_p = clf_p.predict(dataset_orig_test_1.features).reshape(-1,1)
    test_df_copy = copy.deepcopy(dataset_orig_test_1)
    test_df_copy.labels = pred_p
    round_result = measure_final_score(dataset_orig_test_1, test_df_copy, privileged_groups, unprivileged_groups)
    for i in range(len(performance_index)):
        pp_result[performance_index[i]].append(round_result[i])

    clf_f = LogisticRegression()
    clf_f = clf_f.fit(dataset_orig_train_new.features, dataset_orig_train_new.labels)

    test_df_copy = copy.deepcopy(dataset_orig_test_1)
    pred_de1 = clf_p.predict_proba(dataset_orig_test_1.features)
    pred_de2 = clf_f.predict_proba(dataset_orig_test_2.features)

    res = []
    for i in range(len(pred_de1)):
        prob_t = (pred_de1[i][1]+pred_de2[i][1])/2
        if prob_t >= 0.5:
            res.append(1)
        else:
            res.append(0)

    test_df_copy.labels = np.array(res)

    round_result= measure_final_score(dataset_orig_test_1,test_df_copy,privileged_groups,unprivileged_groups)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index + '\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\t%f\n' % (mean(results[p_index]), std(results[p_index])))
for p_index in performance_index:
    fout.write(p_index + '\t')
    for i in range(repeat_time):
        fout.write('%f\t' % pp_result[p_index][i])
    fout.write('%f\t%f\n' % (mean(pp_result[p_index]), std(pp_result[p_index])))
fout.close()

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import os
import argparse
import copy
from WAE import data_dis
import math

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef

def get_data(dataset_used):
    if dataset_used == "adult":
        dataset_orig = AdultDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
    elif dataset_used == "compas":
        dataset_orig = CompasDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
    return dataset_orig


def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf

def measure_final_score(dataset_orig_test, dataset_orig_predict,attr1,attr2):

    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    spd = {}
    aaod = {}
    eod = {}
    for attr in [attr1, attr2]:
        classified_metric_pred = ClassificationMetric(dataset_orig_test, dataset_orig_predict,
                                                      unprivileged_groups=[{attr: 0}],
                                                      privileged_groups=[{attr: 1}])

        spd[attr] = abs(classified_metric_pred.statistical_parity_difference())
        aaod[attr] = classified_metric_pred.average_abs_odds_difference()
        eod[attr] = abs(classified_metric_pred.equal_opportunity_difference())

    return accuracy, recall_macro, precision_macro, f1score_macro, mcc, spd[attr1],aaod[attr1],eod[attr1], spd[attr2],aaod[attr2],eod[attr2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices = ['adult', 'compas'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices = ['rf', 'svm', 'lr'], help="Classifier name")
    parser.add_argument("-p1", "--protected1", type=str, required=True,
                        help="Protected attribute")
    parser.add_argument("-p2", "--protected2", type=str, required=True,
                        help="Protected attribute")

    #adult:sex, race, compas: sex, race

    args = parser.parse_args()
    dataset_used = args.dataset
    attr1 = args.protected1
    attr2 = args.protected2
    clf_name = args.clf

    val_name = "./maat_{}_{}_sexrace.txt".format(clf_name,dataset_used)
    fout = open(val_name, 'w')

    dataset_orig = get_data(dataset_used)

    results = {}
    performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd_for_attr1', 'aod_for_attr1',
                             'eod_for_attr1', 'spd_for_attr2', 'aod_for_attr2', 'eod_for_attr2']
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 50

    for r in range(repeat_time):
        print (r)

        np.random.seed(r)
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        dataset_orig_train_new_for_attr1 = data_dis(pd.DataFrame(dataset_orig_train),attr1,dataset_used)
        dataset_orig_train_new_for_attr2 = data_dis(pd.DataFrame(dataset_orig_train), attr2, dataset_used)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr1)
        dataset_orig_train_new_for_attr1 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr1), columns=dataset_orig.columns)
        dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr2)
        dataset_orig_train_new_for_attr2 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr2),
                                                        columns=dataset_orig.columns)
        dataset_orig_test_3 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                                 protected_attribute_names=[attr1,attr2])
        dataset_orig_train_new_for_attr1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr1,
                                                label_names=['Probability'],
                                                protected_attribute_names=[attr1])
        dataset_orig_train_new_for_attr2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr2,
                                                    label_names=['Probability'],
                                                    protected_attribute_names=[attr2])
        dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                                label_names=['Probability'],
                                                protected_attribute_names=[attr1,attr2])
        dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[attr1])
        dataset_orig_test_3 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_3,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[attr2])

        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator = clf)
        clf1 = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)

        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator = clf)
        clf2 = clf.fit(dataset_orig_train_new_for_attr1.features, dataset_orig_train_new_for_attr1.labels)

        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf3 = clf.fit(dataset_orig_train_new_for_attr2.features, dataset_orig_train_new_for_attr2.labels)

        test_df_copy = copy.deepcopy(dataset_orig_test_1)
        pred_de1 = clf1.predict_proba(dataset_orig_test_1.features)
        pred_de2 = clf2.predict_proba(dataset_orig_test_2.features)
        pred_de3 = clf3.predict_proba(dataset_orig_test_3.features)

        res = []
        for i in range(len(pred_de1)):
            prob_t = (pred_de1[i][1]+pred_de2[i][1]+pred_de3[i][1])/3
            if prob_t >= 0.5:
                res.append(1)
            else:
                res.append(0)

        test_df_copy.labels = np.array(res)

        round_result= measure_final_score(dataset_orig_test_1,test_df_copy,attr1,attr2)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index + '\t')
        for i in range(repeat_time):
            fout.write('%f\t' % results[p_index][i])
        fout.write('%f\t%f\n' % (mean(results[p_index]), std(results[p_index])))
    fout.close()

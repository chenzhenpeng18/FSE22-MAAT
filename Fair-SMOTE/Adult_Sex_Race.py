
# coding: utf-8

import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics
from numpy import mean, std


import sys
sys.path.append(os.path.abspath('..'))

from Generate_Samples import generate_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import argparse
from aif360.datasets import AdultDataset
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
import copy, math

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "rf":
        clf = RandomForestClassifier()
    elif name == "svm":
        clf = LinearSVC()
    return clf

def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)


    spd = {}
    aaod={}
    eod={}
    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['Probability'] = y_pred
    for biased_col in ['sex','race']:
        tt1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=test_df, label_names=['Probability'],
                                 protected_attribute_names=[biased_col])
        tt2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=test_df_copy, label_names=['Probability'],
                                 protected_attribute_names=[biased_col])

        classified_metric_pred = ClassificationMetric(tt1, tt2, unprivileged_groups=[{biased_col: 0}], privileged_groups=[{biased_col: 1}])
        spd[biased_col] = abs(classified_metric_pred.statistical_parity_difference())
        aaod[biased_col] = classified_metric_pred.average_abs_odds_difference()
        eod[biased_col] = abs(classified_metric_pred.equal_opportunity_difference())

    return accuracy,  recall_macro, precision_macro,f1score_macro,  mcc, spd['sex'], aaod['sex'], eod['sex'],spd['race'], aaod['race'], eod['race']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['rf', 'svm', 'lr'], help="Classifier name")

    args = parser.parse_args()
    model_type = args.clf

    ## Load dataset
    dataset_orig = AdultDataset().convert_to_dataframe()[0]
    dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")

    categorical_features = ['sex','race']

    protected_attribute1 = 'sex'
    protected_attribute2 = 'race'

    val_name = "fairsmote_{}_adult_sexrace.txt".format(model_type)
    fout = open(val_name, 'w')

    results = {}
    performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd_for_attr1', 'aod_for_attr1',
                         'eod_for_attr1', 'spd_for_attr2', 'aod_for_attr2', 'eod_for_attr2']
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 50
    for round_num in range(repeat_time):
        print(round_num)

        np.random.seed(round_num)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        # # Find Class & Protected attribute Distribution
        # first one is class value and second one is 'sex' and third one is 'race'
        zero_zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 0)])
        zero_zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 1)])
        zero_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 0)])
        zero_one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 1)])
        one_zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 0)])
        one_zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 1)])
        one_one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 0)])
        one_one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 1)])


        print(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)


        maximum = max(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)
        if maximum == zero_zero_zero:
            print("zero_zero_zero is maximum")
        if maximum == zero_zero_one:
            print("zero_zero_one is maximum")
        if maximum == zero_one_zero:
            print("zero_one_zero is maximum")
        if maximum == zero_one_one:
            print("zero_one_one is maximum")
        if maximum == one_zero_zero:
            print("one_zero_zero is maximum")
        if maximum == one_zero_one:
            print("one_zero_one is maximum")
        if maximum == one_one_zero:
            print("one_one_zero is maximum")
        if maximum == one_one_one:
            print("one_one_one is maximum")

        zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
        zero_zero_one_to_be_incresed = maximum - zero_zero_one
        zero_one_zero_to_be_incresed = maximum - zero_one_zero
        zero_one_one_to_be_incresed = maximum - zero_one_one
        one_zero_zero_to_be_incresed = maximum - one_zero_zero
        one_zero_one_to_be_incresed = maximum - one_zero_one
        one_one_zero_to_be_incresed = maximum - one_one_zero
        one_one_one_to_be_incresed = maximum - one_one_one

        print(zero_zero_zero_to_be_incresed,zero_zero_one_to_be_incresed,zero_one_zero_to_be_incresed,zero_one_one_to_be_incresed,
             one_zero_zero_to_be_incresed,one_zero_one_to_be_incresed,one_one_zero_to_be_incresed,one_one_one_to_be_incresed)


        df_zero_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 0)]
        df_zero_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 1)]
        df_zero_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 0)]
        df_zero_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 1)]
        df_one_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 0)]
        df_one_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 0)
                                               & (dataset_orig_train[protected_attribute2] == 1)]
        df_one_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 0)]
        df_one_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute1] == 1)
                                               & (dataset_orig_train[protected_attribute2] == 1)]

        for cate in categorical_features:
            df_zero_zero_zero[cate] = df_zero_zero_zero[cate].astype(str)
            df_zero_zero_one[cate] = df_zero_zero_one[cate].astype(str)
            df_zero_one_zero[cate] = df_zero_one_zero[cate].astype(str)
            df_zero_one_one[cate] = df_zero_one_one[cate].astype(str)
            df_one_zero_zero[cate] = df_one_zero_zero[cate].astype(str)
            df_one_zero_one[cate] = df_one_zero_one[cate].astype(str)
            df_one_one_zero[cate] = df_one_one_zero[cate].astype(str)
            df_one_one_one[cate] = df_one_one_one[cate].astype(str)


        df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_incresed,df_zero_zero_zero)
        df_zero_zero_one = generate_samples(zero_zero_one_to_be_incresed,df_zero_zero_one)
        df_zero_one_zero = generate_samples(zero_one_zero_to_be_incresed,df_zero_one_zero)
        df_zero_one_one = generate_samples(zero_one_one_to_be_incresed,df_zero_one_one)
        df_one_zero_zero = generate_samples(one_zero_zero_to_be_incresed,df_one_zero_zero)
        df_one_zero_one = generate_samples(one_zero_one_to_be_incresed,df_one_zero_one)
        df_one_one_zero = generate_samples(one_one_zero_to_be_incresed,df_one_one_zero)
        df_one_one_one = generate_samples(one_one_one_to_be_incresed,df_one_one_one)

        # # Append the dataframes
        df = pd.concat([df_zero_zero_zero,df_zero_zero_one,df_zero_one_zero,df_zero_one_one,
        df_one_zero_zero,df_one_zero_one,df_one_one_zero,df_one_one_one])

        for cate in categorical_features:
            df[cate] = df[cate].astype(float)

        # # Check Score after oversampling
        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
        X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        #clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR
        clf = get_classifier(model_type)
        print("-------------------------:Fair-SMOTE-situation")
        round_result = measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test)

        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index + '\t')
        for i in range(repeat_time):
            fout.write('%f\t' % results[p_index][i])
        fout.write('%f\t%f\n' % (mean(results[p_index]), std(results[p_index])))
    fout.close()

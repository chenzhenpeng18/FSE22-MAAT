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
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from aif360.datasets import AdultDataset
from sklearn.preprocessing import MinMaxScaler
from numpy import mean, std
import argparse

import sys
sys.path.append(os.path.abspath('.'))

from Measure_new2 import measure_final_score
from Generate_Samples import generate_samples
from sklearn.calibration import CalibratedClassifierCV

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "rf":
        clf = RandomForestClassifier()
    elif name == "svm":
        clf = LinearSVC()
    return clf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['rf', 'svm', 'lr'], help="Classifier name")
    parser.add_argument("-p", "--protected", type=str, required=True,
                        help="Protected attribute")

    args = parser.parse_args()
    model_type = args.clf
    protected_attribute = args.protected

    #Load dataset
    dataset_orig= AdultDataset().convert_to_dataframe()[0]
    dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")

    categorical_features = ['race','sex']

    val_name = "fairsmotecom_{}_adult_{}.txt".format(model_type, protected_attribute)
    fout = open(val_name, 'w')

    results = {}
    performance_index = ['accuracy', 'recall1', 'recall0', 'recall_macro', 'precision1', 'precision0', 'precision_macro', 'f1score1', 'f1score0', 'f1score_macro', 'roc_auc', 'mcc', 'spd', 'di_old', 'di', 'aod', 'aaod', 'eod','erd']
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 50
    for round_num in range(repeat_time):
        print(round_num)

        np.random.seed(round_num)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle = True)
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train=pd.DataFrame(scaler.transform(dataset_orig_train),columns = dataset_orig.columns)
        dataset_orig_test=pd.DataFrame(scaler.transform(dataset_orig_test),columns = dataset_orig.columns)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        X_train_df_copy = copy.deepcopy(X_train)
        y_train_df_copy = copy.deepcopy(y_train)

        # # Find Class & Protected attribute Distribution
        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        print(zero_zero,zero_one,one_zero,one_one)

        # # Sort these four
        maximum = max(zero_zero,zero_one,one_zero,one_one)
        if maximum == zero_zero:
            print("zero_zero is maximum")
        if maximum == zero_one:
            print("zero_one is maximum")
        if maximum == one_zero:
            print("one_zero is maximum")
        if maximum == one_one:
            print("one_one is maximum")

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        print(zero_zero_to_be_incresed,one_zero_to_be_incresed,one_one_to_be_incresed)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        for cate in categorical_features:
            df_zero_zero[cate] = df_zero_zero[cate].astype(str)
            df_one_zero[cate] = df_one_zero[cate].astype(str)
            df_one_one[cate] = df_one_one[cate].astype(str)

        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero)
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero)
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one)

        # # Append the dataframes
        df = df_zero_zero.append(df_one_zero)
        df = df.append(df_one_one)

        for cate in categorical_features:
            df[cate] = df[cate].astype(float)

        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_zero_one)

        # # Verification
        # first one is class value and second one is protected attribute value
        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])

        print("after smote:", zero_zero, zero_one, one_zero, one_one)


        df = df.reset_index(drop=True)

        # Removal of biased data points using situation testing
        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']

        clf = get_classifier(model_type)
        clf.fit(X_train,y_train)
        removal_list = []

        protected_index = 2  #default:race pay attention here! revision needed!
        if protected_attribute == 'sex':
            protected_index = 3

        for index,row in df.iterrows():
            row_ = [row.values[0:len(row.values)-1]]
            y_normal = clf.predict(row_)
            # Here protected attribute value gets switched
            row_[0][protected_index] = 1 - row_[0][protected_index]
            y_reverse = clf.predict(row_)
            if y_normal[0] != y_reverse[0]:
                removal_list.append(index)

        removal_list = set(removal_list)
        #print(len(removal_list))
        #print(df.shape)
        df_removed = pd.DataFrame(columns=df.columns)

        for index,row in df.iterrows():
            if index in removal_list:
                df_removed = df_removed.append(row, ignore_index=True)
                df = df.drop(index)
        #print(df.shape)

        # first one is class value and second one is protected attribute value
        zero_zero = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 0)])
        zero_one = len(df[(df['Probability'] == 0) & (df[protected_attribute] == 1)])
        one_zero = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 0)])
        one_one = len(df[(df['Probability'] == 1) & (df[protected_attribute] == 1)])
        print("after situation testing:", zero_zero,zero_one,one_zero,one_one)


        # Check Score after Removal
        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
        X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']


        print("-------------------------:Fair-SMOTE-situation")
        round_result = measure_final_score(dataset_orig_test, model_type, X_train, y_train, X_train_df_copy, y_train_df_copy, X_test, y_test, protected_attribute)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index+'\t')
        for i in range(repeat_time):
            fout.write('%f\t' % results[p_index][i])
        fout.write('%f\t%f\n' % (mean(results[p_index]),std(results[p_index])))
    fout.close()


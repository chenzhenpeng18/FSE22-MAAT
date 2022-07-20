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


def data_dis(dataset_orig_test,protected_attribute,dataset_used):

    if dataset_used == 'bank':
        zero_zero = len(dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)])
        zero_one = len(dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)])
        one_zero = len(dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)])
        one_one = len(dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)])
    else:
        zero_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)])
        zero_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)])
        one_zero = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)])
        one_one = len(
            dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)])

    a=zero_one+one_one
    b=-1*(zero_zero*zero_one+2*zero_zero*one_one+one_zero*one_one)
    c=(zero_zero+one_zero)*(zero_zero*one_one-zero_one*one_zero)
    x=(-b-math.sqrt(b*b-4*a*c))/(2*a)
    y=(zero_one+one_one)/(zero_zero+one_zero)*x

    zero_zero_new =int(zero_zero-x)
    one_one_new = int(one_one-y)


    if dataset_used == 'bank':
        zero_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)]
        one_zero_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)]
        zero_zero_set = dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)].sample(n=zero_zero_new)
        one_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)].sample(n=one_one_new)
        new_set = zero_zero_set.append([zero_one_set, one_zero_set, one_one_set], ignore_index=True)
    else:
        zero_one_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)]
        one_zero_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)]
        zero_zero_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)].sample(
            n=zero_zero_new)
        one_one_set = dataset_orig_test[
            (dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)].sample(
            n=one_one_new)
        new_set = zero_zero_set.append([zero_one_set, one_zero_set, one_one_set], ignore_index=True)

    zero_zero = len(new_set[(new_set['Probability'] == 0) & (new_set[protected_attribute] == 0)])
    zero_one = len(new_set[(new_set['Probability'] == 0) & (new_set[protected_attribute] == 1)])
    one_zero = len(new_set[(new_set['Probability'] == 1) & (new_set[protected_attribute] == 0)])
    one_one = len(new_set[(new_set['Probability'] == 1) & (new_set[protected_attribute] == 1)])

    print("WAE_distribution:", zero_zero, zero_one, one_zero, one_one)
    print(zero_zero/(zero_zero+one_zero), zero_one/(zero_one+one_one))


    return new_set
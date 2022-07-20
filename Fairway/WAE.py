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

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "rf":
        clf = RandomForestClassifier()
    elif name == "svm":
        clf = SVC()
    return clf

def data_dis(dataset_orig_test,protected_attribute):

    zero_zero = len(dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)])
    zero_one = len(dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)])
    one_zero = len(dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)])
    one_one = len(dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)])

    '''
    summ = zero_zero+zero_one+one_zero+one_one
    
    uni_prob = set(dataset_orig_test['Probability'])
    print(uni_prob)
    uni_att = set(dataset_orig_test[protected_attribute])
    print(uni_att)
    
    print(zero_zero, zero_one, one_zero, one_one)
    '''
    #print(zero_zero/(zero_zero+one_zero), zero_one/(zero_one+one_one))


    a=zero_one+one_one
    b=-1*(zero_zero*zero_one+2*zero_zero*one_one+one_zero*one_one)
    c=(zero_zero+one_zero)*(zero_zero*one_one-zero_one*one_zero)
    x=(-b-math.sqrt(b*b-4*a*c))/(2*a)
    y=(zero_one+one_one)/(zero_zero+one_zero)*x
    #print("harmo:", int(x), int(y))

    #print("zero_zero:", zero_zero)
    #print("x:", x, y)

    zero_zero_new =int(zero_zero-x)
    one_one_new = int(one_one-y)
    '''
    print("ratio:", (zero_zero+one_zero)/(zero_one+one_one), (zero_zero_new+one_zero)/(zero_one+one_one_new))
    print(zero_zero_new/(zero_zero_new+one_zero))
    print(zero_one/(zero_one+one_one_new))
    '''

    #print('zerozeronew:', zero_zero_new)

    zero_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 1)]
    one_zero_set =  dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 0)]
    zero_zero_set = dataset_orig_test[(dataset_orig_test['Probability'] == 0) & (dataset_orig_test[protected_attribute] == 0)].sample(n=zero_zero_new)
    one_one_set = dataset_orig_test[(dataset_orig_test['Probability'] == 1) & (dataset_orig_test[protected_attribute] == 1)].sample(n=one_one_new)
    new_set = zero_zero_set.append([zero_one_set,one_zero_set,one_one_set],ignore_index=True)


    zero_zero = len(new_set[(new_set['Probability'] == 0) & (new_set[protected_attribute] == 0)])
    zero_one = len(new_set[(new_set['Probability'] == 0) & (new_set[protected_attribute] == 1)])
    one_zero = len(new_set[(new_set['Probability'] == 1) & (new_set[protected_attribute] == 0)])
    one_one = len(new_set[(new_set['Probability'] == 1) & (new_set[protected_attribute] == 1)])

    print("WAR_distribution:", zero_zero, zero_one, one_zero, one_one)
    print(zero_zero/(zero_zero+one_zero), zero_one/(zero_one+one_one))


    return new_set
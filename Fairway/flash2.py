import random
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import sys
import os
sys.path.append(os.path.abspath('.'))

from measure import calculate_recall,calculate_far,calculate_average_odds_difference, calculate_equal_opportunity_difference


def measure_scores(X_train, y_train, X_valid, y_valid, valid_df, biased_col, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    cnf_matrix = confusion_matrix(y_valid, y_pred)

    TN, FP, FN, TP = confusion_matrix(y_valid,y_pred).ravel()

    valid_df_copy = copy.deepcopy(valid_df)
    valid_df_copy['current_pred_' + biased_col] = y_pred

    valid_df_copy['TP_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['TN_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 0) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['FN_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['FP_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 0) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['TP_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['TN_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 0) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['FN_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['FP_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 0) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    a = valid_df_copy['TP_' + biased_col + "_1"].sum()
    b = valid_df_copy['TN_' + biased_col + "_1"].sum()
    c = valid_df_copy['FN_' + biased_col + "_1"].sum()
    d = valid_df_copy['FP_' + biased_col + "_1"].sum()
    e = valid_df_copy['TP_' + biased_col + "_0"].sum()
    f = valid_df_copy['TN_' + biased_col + "_0"].sum()
    g = valid_df_copy['FN_' + biased_col + "_0"].sum()
    h = valid_df_copy['FP_' + biased_col + "_0"].sum()

    recall = calculate_recall(TP, FP, FN, TN)
    far = calculate_far(TP, FP, FN, TN)
    aod = calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    eod = calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)

    return recall, far, aod, eod


def flash_fair_LSR(dataset_orig, biased_col, n_obj):  # biased_col can be "sex" or "race", n_obj can be "ABCD" or "AB" or "CD"
    np.random.seed(100)
    dataset_orig_train, dataset_orig_valid = train_test_split(dataset_orig, test_size=0.3)
    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train[
        'Probability']
    X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid[
        'Probability']

    def convert_lsr(index):  # 30 2 2 100
        a = int(index / 400 + 1)
        b = int(index % 400 / 200 + 1)
        c = int(index % 200 / 100 + 1)
        d = int(index % 100 + 10)
        return a, b, c, d

    all_case = set(range(0, 12000))
    modeling_pool = random.sample(all_case, 20)

    List_X = []
    List_Y = []

    for i in range(len(modeling_pool)):
        temp = convert_lsr(modeling_pool[i])
        List_X.append(temp)
        p1 = temp[0]
        if temp[1] == 1:
            p2 = 'l1'
        else:
            p2 = 'l2'
        if temp[2] == 1:
            p3 = 'liblinear'
        else:
            p3 = 'saga'
        p4 = temp[3]
        model = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)

        all_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col, model)
        four_goal = all_value[0] + all_value[1] + all_value[2] + all_value[3]
        two_goal_recall_far = all_value[0] + all_value[1]
        two_goal_aod_eod = all_value[2] + all_value[3]
        if n_obj == "ABCD":
            List_Y.append(four_goal)
        elif n_obj == "AB":
            List_Y.append(two_goal_recall_far)
        elif n_obj == "CD":
            List_Y.append(two_goal_aod_eod)
        else:
            print("Wrong number of objects")

    remain_pool = all_case - set(modeling_pool)
    test_list = []
    for i in list(remain_pool):
        test_list.append(convert_lsr(i))

    upper_model = DecisionTreeRegressor()
    life = 20

    while len(List_X) < 200 and life > 0:
        upper_model.fit(List_X, List_Y)
        candidate = random.sample(test_list, 1)
        test_list.remove(candidate[0])
        candi_pred_value = upper_model.predict(candidate)
        if candi_pred_value < np.median(List_Y):
            List_X.append(candidate[0])
            candi_config = candidate[0]

            pp1 = candi_config[0]
            if candi_config[1] == 1:
                pp2 = 'l1'
            else:
                pp2 = 'l2'
            if candi_config[2] == 1:
                pp3 = 'liblinear'
            else:
                pp3 = 'saga'
            pp4 = candi_config[3]

            candi_model = LogisticRegression(C=pp1, penalty=pp2, solver=pp3, max_iter=pp4)
            candi_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col,
                                         candi_model)
            candi_four_goal = candi_value[0] + candi_value[1] + candi_value[2] + candi_value[3]
            candi_two_goal_recall_far = candi_value[0] + candi_value[1]
            candi_two_goal_aod_eod = candi_value[2] + candi_value[3]
            if n_obj == "ABCD":
                List_Y.append(candi_four_goal)
            elif n_obj == "AB":
                List_Y.append(candi_two_goal_recall_far)
            elif n_obj == "CD":
                List_Y.append(candi_two_goal_aod_eod)
        else:
            life -= 1

    min_index = int(np.argmin(List_Y))

    best_config = List_X[min_index]
    print("best_config", best_config)
    p1 = best_config[0]
    if best_config[1] == 1:
        p2 = 'l1'
    else:
        p2 = 'l2'
    if best_config[2] == 1:
        p3 = 'liblinear'
    else:
        p3 = 'saga'
    p4 = best_config[3]
    clf = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)
    return clf


def flash_fair_SVC(dataset_orig, biased_col, n_obj):
    np.random.seed(100)
    dataset_orig_train, dataset_orig_valid = train_test_split(dataset_orig, test_size=0.3)
    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']

    #C=0.000001,0.00001,0.0001,0.001,0.01, 0.1, 0, 1, 10, 100, 1000, 10000, 100000, 1000000 https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
    def convert_svc(index):
        a = int(index / 1000) #0-12
        return a,a

    all_case = set(range(0, 13000))
    modeling_pool = random.sample(all_case, 20)

    List_X = []
    List_Y = []

    svc_c_list = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

    for i in range(len(modeling_pool)):
        temp = convert_svc(modeling_pool[i])
        List_X.append(temp)
        p2  = svc_c_list[temp[1]]
        model = LinearSVC(C=p2)

        all_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col, model)
        four_goal = all_value[0] + all_value[1] + all_value[2] + all_value[3]
        two_goal_recall_far = all_value[0] + all_value[1]
        two_goal_aod_eod = all_value[2] + all_value[3]
        if n_obj == "ABCD":
            List_Y.append(four_goal)
        elif n_obj == "AB":
            List_Y.append(two_goal_recall_far)
        elif n_obj == "CD":
            List_Y.append(two_goal_aod_eod)
        else:
            print("Wrong number of objects")

    remain_pool = all_case - set(modeling_pool)
    test_list = []
    for i in list(remain_pool):
        test_list.append(convert_svc(i))

    upper_model = DecisionTreeRegressor()
    life = 20

    while len(List_X) < 200 and life > 0:
        upper_model.fit(List_X, List_Y)
        candidate = random.sample(test_list, 1)
        test_list.remove(candidate[0])
        candi_pred_value = upper_model.predict(candidate)
        if candi_pred_value < np.median(List_Y):
            List_X.append(candidate[0])
            candi_config = candidate[0]

            pp2 = svc_c_list[candi_config[1]]

            candi_model = LinearSVC(C=pp2)
            candi_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col,
                                         candi_model)
            candi_four_goal = candi_value[0] + candi_value[1] + candi_value[2] + candi_value[3]
            candi_two_goal_recall_far = candi_value[0] + candi_value[1]
            candi_two_goal_aod_eod = candi_value[2] + candi_value[3]
            if n_obj == "ABCD":
                List_Y.append(candi_four_goal)
            elif n_obj == "AB":
                List_Y.append(candi_two_goal_recall_far)
            elif n_obj == "CD":
                List_Y.append(candi_two_goal_aod_eod)
        else:
            life -= 1

    min_index = int(np.argmin(List_Y))

    best_config = List_X[min_index]
    print("best_config", best_config)
    p2 = svc_c_list[best_config[1]]
    clf = CalibratedClassifierCV(base_estimator=LinearSVC(C=p2))
    return clf

def flash_fair_RF(dataset_orig, biased_col, n_obj):
    np.random.seed(100)
    dataset_orig_train, dataset_orig_valid = train_test_split(dataset_orig, test_size=0.3)
    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']

    #n_estimators = [100, 300, 500, 800, 1200], max_depth = [5, 8, 15, 25, 30], min_samples_split = [2, 5, 10, 15, 100], min_samples_leaf = [1, 2, 5, 10], https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6

    def convert_rf(index):  # 30 2 2 100
        a = int(index / 2400) #0-4
        b = int(index % 400 / 80) #0-4
        c = int(index % 200 / 40) #0-4
        d = int(index % 100 / 25) #0-3
        return a, b, c, d

    all_case = set(range(0, 12000))
    modeling_pool = random.sample(all_case, 20)

    List_X = []
    List_Y = []

    rf_n_estimators_list = [100, 300, 500, 800, 1200]
    rf_max_depth_list = [5, 8, 15, 25, 30]
    rf_min_samples_split_list = [2, 5, 10, 15, 100]
    rf_min_samples_leaf_list = [1, 2, 5, 10]

    for i in range(len(modeling_pool)):
        temp = convert_rf(modeling_pool[i])
        List_X.append(temp)
        p1 = rf_n_estimators_list[temp[0]]
        p2  = rf_max_depth_list[temp[1]]
        p3 = rf_min_samples_split_list[temp[2]]
        p4 =  rf_min_samples_leaf_list[temp[3]]
        model = RandomForestClassifier(n_estimators=p1, max_depth=p2, min_samples_split = p3, min_samples_leaf=p4)

        all_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col, model)
        four_goal = all_value[0] + all_value[1] + all_value[2] + all_value[3]
        two_goal_recall_far = all_value[0] + all_value[1]
        two_goal_aod_eod = all_value[2] + all_value[3]
        if n_obj == "ABCD":
            List_Y.append(four_goal)
        elif n_obj == "AB":
            List_Y.append(two_goal_recall_far)
        elif n_obj == "CD":
            List_Y.append(two_goal_aod_eod)
        else:
            print("Wrong number of objects")

    remain_pool = all_case - set(modeling_pool)
    test_list = []
    for i in list(remain_pool):
        test_list.append(convert_rf(i))

    upper_model = DecisionTreeRegressor()
    life = 20

    while len(List_X) < 200 and life > 0:
        upper_model.fit(List_X, List_Y)
        candidate = random.sample(test_list, 1)
        test_list.remove(candidate[0])
        candi_pred_value = upper_model.predict(candidate)
        if candi_pred_value < np.median(List_Y):
            List_X.append(candidate[0])
            candi_config = candidate[0]

            pp1 = rf_n_estimators_list[candi_config[0]]
            pp2 = rf_max_depth_list[candi_config[1]]
            pp3 = rf_min_samples_split_list[candi_config[2]]
            pp4 = rf_min_samples_leaf_list[candi_config[3]]

            candi_model = RandomForestClassifier(n_estimators=pp1, max_depth=pp2, min_samples_split = pp3, min_samples_leaf=pp4)
            candi_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col,
                                         candi_model)
            candi_four_goal = candi_value[0] + candi_value[1] + candi_value[2] + candi_value[3]
            candi_two_goal_recall_far = candi_value[0] + candi_value[1]
            candi_two_goal_aod_eod = candi_value[2] + candi_value[3]
            if n_obj == "ABCD":
                List_Y.append(candi_four_goal)
            elif n_obj == "AB":
                List_Y.append(candi_two_goal_recall_far)
            elif n_obj == "CD":
                List_Y.append(candi_two_goal_aod_eod)
        else:
            life -= 1

    min_index = int(np.argmin(List_Y))

    best_config = List_X[min_index]
    print("best_config", best_config)
    p1 = rf_n_estimators_list[best_config[0]]
    p2 = rf_max_depth_list[best_config[1]]
    p3 = rf_min_samples_split_list[best_config[2]]
    p4 = rf_min_samples_leaf_list[best_config[3]]
    clf = RandomForestClassifier(n_estimators=p1, max_depth=p2, min_samples_split = p3, min_samples_leaf=p4)
    return clf

def flash_fair(dataset_orig, biased_col, model_type, n_obj="ABCD"):
    if model_type == "lr":
        return flash_fair_LSR(dataset_orig, biased_col, n_obj)
    elif model_type == "rf":
        return flash_fair_RF(dataset_orig, biased_col, n_obj)
    elif model_type == "svm":
        return flash_fair_SVC(dataset_orig, biased_col, n_obj)
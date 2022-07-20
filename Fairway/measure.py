import copy
import math

from sklearn.metrics import confusion_matrix,classification_report
import numpy as np

def get_counts(clf, x_train, y_train, x_test, y_test, test_df, biased_col, metric='aod'):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    print(cnf_matrix)
    # print(classification_report(y_test, y_pred))

    TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['current_pred_'  + biased_col] = y_pred

    test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 1) &
                                           (test_df_copy['current_pred_' + biased_col] == 1) &
                                           (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    a = test_df_copy['TP_' + biased_col + "_1"].sum()
    b = test_df_copy['TN_' + biased_col + "_1"].sum()
    c = test_df_copy['FN_' + biased_col + "_1"].sum()
    d = test_df_copy['FP_' + biased_col + "_1"].sum()
    e = test_df_copy['TP_' + biased_col + "_0"].sum()
    f = test_df_copy['TN_' + biased_col + "_0"].sum()
    g = test_df_copy['FN_' + biased_col + "_0"].sum()
    h = test_df_copy['FP_' + biased_col + "_0"].sum()

    if metric=='aod':
        return  calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric=='eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    # elif metric=='d2h':
    #     return d2h(TP,FP,FN,TN)
    elif metric=='recall':
        return calculate_recall(TP,FP,FN,TN)
    elif metric=='far':
        return calculate_far(TP,FP,FN,TN)

def get_counts_german(clf, x_train, y_train, x_test, y_test, test_df, biased_col, metric='aod'):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    # print(cnf_matrix)
    # print(classification_report(y_test, y_pred))

    TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['current_pred_'  + biased_col] = y_pred

    test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 1) &
                                           (test_df_copy['current_pred_' + biased_col] == 1) &
                                           (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 2) &
                                                  (test_df_copy['current_pred_' + biased_col] == 2) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 2) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 2) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 2) &
                                                  (test_df_copy['current_pred_' + biased_col] == 2) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 2) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 2) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    a = test_df_copy['TP_' + biased_col + "_1"].sum()
    b = test_df_copy['TN_' + biased_col + "_1"].sum()
    c = test_df_copy['FN_' + biased_col + "_1"].sum()
    d = test_df_copy['FP_' + biased_col + "_1"].sum()
    e = test_df_copy['TP_' + biased_col + "_0"].sum()
    f = test_df_copy['TN_' + biased_col + "_0"].sum()
    g = test_df_copy['FN_' + biased_col + "_0"].sum()
    h = test_df_copy['FP_' + biased_col + "_0"].sum()

    if metric=='aod':
        return  calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric=='eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    # elif metric=='d2h':
    #     return d2h(TP,FP,FN,TN)
    elif metric=='recall':
        return calculate_recall(TP,FP,FN,TN)
    elif metric=='far':
        return calculate_far(TP,FP,FN,TN)


def calculate_average_odds_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
	TPR_male = TP_male/(TP_male+FN_male)
	TPR_female = TP_female/(TP_female+FN_female)
	FPR_male = FP_male/(FP_male+TN_male)
	FPR_female = FP_female/(FP_female+TN_female)
	average_odds_difference = abs(abs(TPR_male - TPR_female) + abs(FPR_male - FPR_female))/2
	#print("average_odds_difference",average_odds_difference)
	return average_odds_difference


def calculate_equal_opportunity_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
	TPR_male = TP_male/(TP_male+FN_male)
	TPR_female = TP_female/(TP_female+FN_female)
	equal_opportunity_difference = abs(TPR_male - TPR_female)
	#print("equal_opportunity_difference:",equal_opportunity_difference)
	return equal_opportunity_difference

def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
      recall = 0
    # To maximize recall, we will return ( 1 - recall)
    return 1 - recall

def calculate_far(TP,FP,FN,TN):
    if (FP + TN) != 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return far


## Calculate d2h
def d2h(TP,FP,FN,TN):
   if (FP + TN) != 0:
       far = FP/(FP+TN)
   if (TP + FN) != 0:
       recall = TP/(TP + FN)
   dist2heaven = math.sqrt((1 - (recall)**2)/2)
   #print("dist2heaven:",dist2heaven)
   return dist2heaven


def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_col, metric):
    df = copy.deepcopy(test_df)
    return get_counts(clf, X_train, y_train, X_test, y_test, df, biased_col, metric=metric)

def measure_final_score_german(test_df, clf, X_train, y_train, X_test, y_test, biased_col, metric):
    df = copy.deepcopy(test_df)
    return get_counts_german(clf, X_train, y_train, X_test, y_test, df, biased_col, metric=metric)
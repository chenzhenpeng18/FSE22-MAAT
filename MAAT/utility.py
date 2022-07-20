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

# protected in {sex,race,age}
def get_data(dataset_used, protected,preprocessed = False):
    if dataset_used == "adult":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = AdultDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
    elif dataset_used == "german":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        dataset_orig = GermanDataset().convert_to_dataframe()[0]
        dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
        dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
    elif dataset_used == "compas":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = CompasDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
        #dataset_orig['Probability'] = 1 - dataset_orig['Probability']  # make favorable_class as 1
    elif dataset_used == "bank":
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = BankDataset().convert_to_dataframe()[0]
        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)
        #dataset_orig['age'] = 1 - dataset_orig['age']
    elif dataset_used == "mep":
        privileged_groups = [{'RACE': 1}]
        unprivileged_groups = [{'RACE': 0}]
        dataset_orig = MEPSDataset19().convert_to_dataframe()[0]
        dataset_orig.rename(columns={'UTILIZATION': 'Probability'}, inplace=True)
    return dataset_orig, privileged_groups,unprivileged_groups


def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf

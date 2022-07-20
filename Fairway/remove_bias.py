import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
sys.path.append(os.path.abspath('.'))


def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "rf":
        clf = RandomForestClassifier()
    elif name == "svm":
        clf = LinearSVC()
    return clf

def remove_bias(dataset_orig_train, protected_attribute, model_type):
    # divide training data based on protected_attribute
    dataset_orig_privileged, dataset_orig_unprivileged = [x for _, x in dataset_orig_train.groupby(dataset_orig_train[protected_attribute] == 0)]

    #print("privi in training:", dataset_orig_privileged.shape)
    #print("unprivi in training:", dataset_orig_unprivileged.shape)

    # Train the model for the privileged group
    X_train_privileged, y_train_privileged = dataset_orig_privileged.loc[:, dataset_orig_privileged.columns != 'Probability'], dataset_orig_privileged['Probability']
    X_train_privileged = X_train_privileged.loc[:,X_train_privileged.columns != protected_attribute]
    clf_privileged = get_classifier(model_type)
    clf_privileged.fit(X_train_privileged, y_train_privileged)

    # Train the model for the unprivileged group
    X_train_unprivileged, y_train_unprivileged = dataset_orig_unprivileged.loc[:,dataset_orig_unprivileged.columns != 'Probability'], dataset_orig_unprivileged['Probability']
    X_train_unprivileged = X_train_unprivileged.loc[:, X_train_unprivileged.columns != protected_attribute]
    clf_unprivileged = get_classifier(model_type)
    clf_unprivileged.fit(X_train_unprivileged, y_train_unprivileged)

    # Remove biased rows
    for index, row in dataset_orig_train.iterrows():
        row_ = [row[(row.index!=protected_attribute) & (row.index!='Probability')]]
        y_privileged = clf_privileged.predict(row_)
        y_unprivileged = clf_unprivileged.predict(row_)
        if y_privileged[0] != y_unprivileged[0]:
            dataset_orig_train = dataset_orig_train.drop(index)

    return dataset_orig_train
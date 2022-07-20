from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
# protected in {sex,race,age}
def get_data(dataset_used, protected,preprocessed = False):
    """ Obtains dataset from AIF360.

    Parameters:
        dataset_used (str) -- Name of the dataset
        protected (str)    -- Protected attribute used
    Returns:
        dataset_orig (dataset)     -- Classifier with default configuration from scipy
        privileged_groups (list)   -- Attribute and corresponding value of privileged group 
        unprivileged_groups (list) -- Attribute and corresponding value of unprivileged group 
        optim_options (dict)       -- Options if provided by AIF360
    """
    if dataset_used == "adult":
        mutation_strategy  = {"0":[1,0]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_adult(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_adult(['race'])
            
        optim_options = {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        if not preprocessed:
            dataset_orig = AdultDataset()
    elif dataset_used == "german":
        mutation_strategy = {"1": [0, 1]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_german(['sex'])
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
            dataset_orig = load_preproc_data_german(['age'])
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.1,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }    
        if not preprocessed:
            dataset_orig = GermanDataset()
    elif dataset_used == "compas":
        mutation_strategy = {"0": [1, 0]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_compas(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_compas(['race'])
            
        optim_options = {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        if not preprocessed:
            dataset_orig = CompasDataset()
    elif dataset_used == "bank":
        mutation_strategy = {"0": [1, 0]}
        privileged_groups = [{'age': 1}]  
        unprivileged_groups = [{'age': 0}]
        dataset_orig = BankDataset()
        #dataset_orig.features[:,0] =  dataset_orig.features[:,0]>=25
        optim_options = None
    elif dataset_used == "mep":
        mutation_strategy = {"0": [1, 0]}
        privileged_groups = [{'RACE': 1}]
        unprivileged_groups = [{'RACE': 0}]
        dataset_orig = MEPSDataset19()
        optim_options = None
    return dataset_orig, privileged_groups,unprivileged_groups,optim_options,mutation_strategy

def write_to_file(fname,content):
    """ Write content into a line of a file.

    Parameters:
        fname (str)   -- Name of file to write to 
        content (str) -- Line that is appendend to file
    """
    f = open(fname, "a")
    f.write(content)
    f.write("\n")
    f.close()



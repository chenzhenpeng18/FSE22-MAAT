import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Fairea.utility import get_data,write_to_file
from Fairea.fairea import create_baseline,normalize,get_classifier,classify_region,compute_area
from shapely.geometry import Polygon, Point, LineString
from matplotlib import pyplot as plt
import matplotlib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'german', 'compas', 'bank', 'mep'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected attribute")

args = parser.parse_args()
dataset_used = args.dataset
attr = args.protected
clf_name = args.clf

degrees = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
dataset_orig, privileged_groups,unprivileged_groups,optim_options, mutation_strategies = get_data(dataset_used, attr)

fout = open('../Fairea_baseline/'+dataset_used+'_'+clf_name+'_'+attr+'_baseline','w')
res = create_baseline(clf_name,dataset_orig, privileged_groups,unprivileged_groups,
                    data_splits=50,repetitions=50,odds=mutation_strategies,options = [0,1],
                   degrees = degrees)
per_list = ['accuracy', 'precision', 'recall', 'f1score', 'mcc']
fair_list = ['spd', 'aod','eod']

res_forplot = {}
index = '0'
if dataset_used == 'german':
    index = '1'
for i in range(len(per_list)):
    res_forplot[per_list[i]]= {}
    res_forplot[per_list[i]][index] = np.array([np.mean([row[3+i] for row in res[index][degree]]) for degree in degrees])

for i in range(len(fair_list)):
    res_forplot[fair_list[i]]= {}
    res_forplot[fair_list[i]][index] = np.array([np.mean([row[i] for row in res[index][degree]]) for degree in degrees])

for i in per_list:
    fout.write(i)
    for numnum in res_forplot[i][index]:
        fout.write('\t%f' % numnum)
    fout.write('\n')

for i in fair_list:
    fout.write(i)
    for numnum in res_forplot[i][index]:
        fout.write('\t%f' % numnum)
    fout.write('\n')

fout.close()

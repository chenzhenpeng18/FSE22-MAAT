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

base_points = {}
for i in ['lr']:
    base_points[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
        base_points[i][j]={}

base_map = {1:'Acc', 2: 'Mac-P', 3: 'Mac-R', 4: 'Mac-F1', 5: 'MCC', 6: 'SPD', 7: 'AOD', 8:'EOD'}
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
    (dataset_pre,dataset_aft) = dataset.lower().split('-')
    if dataset  == 'Mep_Race':
        dataset_pre = 'mep'
        dataset_aft = 'RACE'
    for i in ['lr']:
        fin = open('../Fairea_baseline/'+dataset_pre+'_'+i+'_'+dataset_aft+'_baseline','r')
        count = 0
        for line in fin:
            count += 1
            if count in base_map:
                base_points[i][dataset][base_map[count]] = np.array(list(map(float,line.strip().split('\t')[1:])))
        fin.close()

data = {}
for i in ['lr']:
    data[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
        data[i][j]={}
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC','SPD','AOD','EOD']:
            data[i][j][k]={}

data_key_value_used = {1:'Acc', 2:'Mac-R', 3:'Mac-P', 4:'Mac-F1', 5:'MCC', 6:'SPD', 7:'AOD', 8:'EOD'}
for name in ['lr','svm','rf','bayes','dt']:
    for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
        (dataset_pre,dataset_aft) = dataset.lower().split('-')
        if dataset == 'Mep-Race':
            dataset_pre = 'mep'
            dataset_aft = 'RACE'
        fin = open('../RQ3_results/maat_lrfairmodel_'+name+'_'+dataset_pre+'_'+dataset_aft+'.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                data['lr'][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:51]))
        fin.close()

region_count = {}
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
    region_count[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in ['lr']:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in ['lr','svm','rf','bayes','dt']:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in ['lr']:
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
        for fairmetric in ['SPD','AOD','EOD']:
            for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
                for name in ['lr','svm','rf','bayes','dt']:
                    methods = dict()
                    name_fair50 = data[i][j][fairmetric][name]
                    name_per50 = data[i][j][permetric][name]
                    for count in range(50):
                        methods[str(count)] = (float(name_per50[count]), float(name_fair50[count]))
                    normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                    baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                    mitigation_regions = classify_region(baseline, normalized_methods)
                    for count in mitigation_regions:
                        region_count[j][fairmetric][permetric][i][name][mitigation_regions[count]]+=1


perfor = {}
for i in ['lr']:
    perfor[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
        perfor[i][j]={}
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            perfor[i][j][k]={}

data_key_value_used = {9:'Acc', 10:'Mac-R', 11:'Mac-P', 12:'Mac-F1', 13:'MCC'}
for j in ['lr']:
    for name in ['lr','svm','rf','bayes','dt']:
        for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
            (dataset_pre,dataset_aft) = dataset.lower().split('-')
            if dataset == 'Mep-Race':
                dataset_pre = 'mep'
                dataset_aft = 'RACE'
            fin = open('../RQ3_results/maat_lrfairmodel_'+name+'_'+dataset_pre+'_'+dataset_aft+'.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    perfor[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:51]))
            fin.close()

fout = open('figure4_results','w')
fout.write('\tAccuracy\n')
for name in ['lr','svm','rf','bayes','dt']:
    fout.write(name)
    summ = []
    for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex',
              'Bank-Age', 'Mep-Race']:
        summ +=  perfor['lr'][j]['Acc'][name]
    fout.write('\t%f\n' % np.mean(summ))


fout.write('\tProportions of cases beating the trade-off baseline\n')

for name in ['lr','svm','rf','bayes','dt']:
    fout.write(name)
    final_sum = 0
    final_count = {}
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for fairmetric in ['SPD', 'AOD', 'EOD']:
        for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
            for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex',
                          'Bank-Age', 'Mep-Race']:
                for i in ['lr']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    fout.write('\t%f' % ((final_count['win-win']+final_count['good'])/final_sum))
    fout.write('\n')
fout.close()

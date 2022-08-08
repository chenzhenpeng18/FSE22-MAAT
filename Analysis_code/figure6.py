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
for i in ['rf','lr','svm']:
    base_points[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
        base_points[i][j]={}

base_map = {1:'Acc', 2: 'Mac-P', 3: 'Mac-R', 4: 'Mac-F1', 5: 'MCC', 6: 'SPD', 7: 'AOD', 8:'EOD'}
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
    (dataset_pre,dataset_aft) = dataset.lower().split('-')
    if dataset  == 'Mep_Race':
        dataset_pre = 'mep'
        dataset_aft = 'RACE'
    for i in ['rf', 'lr', 'svm']:
        fin = open('../Fairea_baseline/'+dataset_pre+'_'+i+'_'+dataset_aft+'_baseline','r')
        count = 0
        for line in fin:
            count += 1
            if count in base_map:
                base_points[i][dataset][base_map[count]] = np.array(list(map(float,line.strip().split('\t')[1:])))
        fin.close()

data = {}
for i in ['rf','lr','svm']:
    data[i]={}
    for j in ['adult','compas']:
        data[i][j]={}
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC','SPD1','AOD1','EOD1','SPD2','AOD2','EOD2']:
            data[i][j][k]={}

data_key_value_used = {1:'Acc', 2:'Mac-R', 3:'Mac-P', 4:'Mac-F1', 5:'MCC', 6:'SPD1', 7:'AOD1', 8:'EOD1',9:'SPD2',10:'AOD2',11:'EOD2'}
for j in ['lr','rf','svm']:
    for dataset in ['adult','compas']:
        for name in ['maat','fairsmote']:
            fin = open('../RQ5_results/'+name+'_'+j+'_'+dataset+'_sexrace.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:51]))
            fin.close()

region_count = {}
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race']:
    region_count[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in ['rf','lr','svm']:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in ['maat','fairsmote']:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in ['rf','lr','svm']:
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race']:
        if j in ['Adult-Sex','Adult-Race']:
            zz = 'adult'
        else:
            zz = 'compas'
        for fairmetric in ['SPD','AOD','EOD']:
            if j in ['Adult-Sex', 'Compas-Sex']:
                mm = fairmetric+'1'
            else:
                mm = fairmetric+'2'
            for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
                for name in ['maat','fairsmote']:
                    methods = dict()
                    name_fair50 = data[i][zz][mm][name]
                    name_per50 = data[i][zz][permetric][name]
                    for count in range(50):
                        methods[str(count)] = (float(name_per50[count]), float(name_fair50[count]))
                    normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                    baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                    mitigation_regions = classify_region(baseline, normalized_methods)
                    for count in mitigation_regions:
                        region_count[j][fairmetric][permetric][i][name][mitigation_regions[count]]+=1

fout = open('figure6_results','w')
for region_kind in ['win-win', 'good', 'inverted', 'poor', 'lose-lose']:
    fout.write('\t'+region_kind)
fout.write('\n')

for name in ['fairsmote','maat']:
    for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race']:
        fout.write(name +'_'+ j)
        final_count = {}
        for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                final_count[region_kind] = 0
        for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
            for fairmetric in ['SPD', 'AOD', 'EOD']:
                for i in ['rf', 'lr', 'svm']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
        final_sum = 0
        for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
            final_sum += final_count[region_kind]
        for region_kind in ['win-win', 'good', 'inverted', 'bad', 'lose-lose']:
            fout.write('\t%f' % (final_count[region_kind] / final_sum))
        fout.write('\n')
fout.close()
from numpy import mean, std,sqrt
import scipy.stats as stats

def mann(x,y):
	return stats.mannwhitneyu(x,y)[1]

data = {}
for i in ['rf','lr','svm']:
    data[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
        data[i][j]={}
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC','SPD','AOD','EOD']:
            data[i][j][k]={}

data_key_value_used = {1:'Acc', 2:'Mac-R', 3:'Mac-P', 4:'Mac-F1', 5:'MCC', 6:'SPD', 7:'AOD', 8:'EOD'}
for j in ['lr','rf','svm']:
    for name in ['default', 'rew','adv','roc','fairway','fairsmote','maat']:
        for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
            (dataset_pre,dataset_aft) = dataset.lower().split('-')
            if dataset == 'Mep-Race':
                dataset_pre = 'mep'
                dataset_aft = 'RACE'
            fin = open('../RQ1&2_results/'+name+'_'+j+'_'+dataset_pre+'_'+dataset_aft+'.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:51]))
            fin.close()

Pdegrade_ratio = {}
Fincrease_ratio = {}
for name in ['rew','adv','roc','fairway','fairsmote','maat']:
    Pdegrade_ratio[name] = 0
    Fincrease_ratio[name] = 0

for z in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','Bank-Age','Mep-Race']:
    for j in ['lr','rf','svm']:
        for k in ['SPD', 'AOD', 'EOD']:
            default_list = data[j][z][k]['default']
            default_valuefork = mean(default_list)
            for name in ['rew','adv','roc','fairway','fairsmote','maat']:
                real_list = data[j][z][k][name]
                real_valuefork = mean(real_list)
                if mann(default_list, real_list) < 0.05:
                    if real_valuefork < default_valuefork:
                        Fincrease_ratio[name]+=1
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            default_list = data[j][z][k]['default']
            default_valuefork = mean(default_list)
            for name in ['rew', 'adv', 'roc', 'fairway', 'fairsmote', 'maat']:
                real_list = data[j][z][k][name]
                real_valuefork = mean(real_list)
                if mann(default_list, real_list) < 0.05:
                    if real_valuefork < default_valuefork:
                        Pdegrade_ratio[name] += 1

fout = open('table2_result','w')
for name in ['rew', 'adv',  'roc', 'fairway', 'fairsmote', 'maat']:
    fout.write('\t'+name)
fout.write('\n')
fout.write('Fairness increases')
for name in ['rew', 'adv',  'roc', 'fairway', 'fairsmote', 'maat']:
    fout.write('\t%f' % (float(Fincrease_ratio[name])/(3*7*3)))
fout.write('\n')
fout.write('Performance decreases')
for name in ['rew', 'adv', 'roc', 'fairway', 'fairsmote', 'maat']:
    fout.write('\t%f' % (float(Pdegrade_ratio[name]) / (3 * 7 * 5)))
fout.write('\n')
fout.close()

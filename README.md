# [ESEC/FSE 2022] MAAT: A Novel Ensemble Approach to Addressing Fairness and Performance Bugs for Machine Learning Software

Welcome to visit the homepage of our ESEC/FSE'22 paper entitled "MAAT: A Novel Ensemble Approach to Addressing Fairness and Performance Bugs for Machine Learning Software". The homepage contains the source code of MAAT and other existing bias mitigation methods that we use in our paper, as well as the intermediate results, the installation instructions, and a replication guideline.

## Experimental environment

We use Python 3.7 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit for implementing bias mitigation methods and computing fairness metrics. 

Installation instructions for Python 3.7 and AIF360 can be found on https://github.com/Trusted-AI/AIF360. That page provides several ways for the installation. We recommend creating a virtual environment for it (as shown below), because AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. If you would like to try other installation ways or encounter any errors during the installation proces, please refer to the page (https://github.com/Trusted-AI/AIF360) for help.

#### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.7 environment, run:

```bash
conda create --name aif360 python=3.7
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source
deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install 'aif360'
```

[comment]: <> (This toolkit can be installed as follows:)

[comment]: <> (```)

[comment]: <> (pip install aif360)

[comment]: <> (```)

[comment]: <> (More information on installing AIF360 can be found on https://github.com/Trusted-AI/AIF360.)

In addition, we require the following Python packages. Note that TensorFlow is only required for implementing the existing bias mitigation method named ADV. If you do not want to implement this method, you can skip the installation of TensorFlow (the last line of the following commands).
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
pip install --upgrade protobuf==3.20.0
pip install fairlearn
```

## Dataset

We use the five default datasets supported by the AIF360 toolkit. **When running the scripts that invoke these datasets, you will be prompted how to download these datasets and in which folders they need to be placed.** You can also refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files.

## Scripts and results
The repository contains the following folders:

* ```MAAT/``` contains code for implementing our approach.
  
* ```Fair360/``` contains code for implementing three bias mitigation methods (i.e., REW, ADV, and ROC) in AIF360.

* ```Fairway/``` contains code for implementing Fairway, a bias mitigation method proposed by [Chakraborty et al.](https://doi.org/10.1145/3368089.3409697) in ESEC/FSE 2020.

* ```Fair-SMOTE/``` contains code for implementing Fair-SMOTE, a bias mitigation method proposed by [Chakraborty et al.](https://doi.org/10.1145/3468264.3468537) in ESEC/FSE 2021.

* ```Fairea/``` contains code for benchmarking fairness-performance trade-off. It is implemented based on Fairea, a trade-off benchmark method proposed by [Hort et al.](https://doi.org/10.1145/3468264.3468565) in ESEC/FSE 2021.

* ```Analysis_code/``` contains code for calculating the baselines for Fairea and the code for analyzing the effectiveness distribution.
  
* ```Fairea_baseline/``` contains the performance and fairness metric values of the data points that construct the Fairea baselines.

* ```RQ1&2_results/```, ```RQ3_results/```, ```RQ4_results/```, and ```RQ5_results/``` contain the raw results of the models after applying different bias mitigation methods/strategies. Each file in these folders has 53 columns, with the first column indicating the metric, the next 50 columns the metric values of 50 runs, and the last two columns the mean and std values of the 50 runs.

## Reproduction 
You can reproduce all the results for all our research questions (RQs) based on the intermediate results provided by us.

### RQ1 (Trade-off Effectiveness)
The results of RQ1 are shown in Figure 2 and Table 2. You can reproduce the results as follows:
```
cd Analysis_code
python figure2.py
python table2.py
```

### RQ2 (Applicability)
The results of RQ2 are shown in Figure 3. You can reproduce the results as follows:
```
cd Analysis_code
python figure3.py
```

### RQ3 (Influence of Fairness and Performance Models)
The results of RQ3 are shown in Table 3 and Figure 4. You can reproduce the results as follows:
```
cd Analysis_code
python table3.py
python figure4.py
```

### RQ4 (Influence of Combination Strategies)
The results of RQ4 are shown in Figure 5. You can reproduce the results as follows:
```
cd Analysis_code
python figure5.py
```

### RQ5 (Multiple Protected Attributes)
The results of RQ5 are shown in Figure 6. You can reproduce the results as follows:
```
cd Analysis_code
python figure6.py
```

## Step-by-step Guide
You can also reproduce the results from scratch. We provide the step-by-step guide on how to reproduce the intermediate results and obtain the results for RQs based on them.

### RQ1 (Trade-off Effectiveness): What fairness-performance trade-off does MAAT achieve?
This RQ compares MAAT with existing bias mitigation methods by analyzing which trade-off effectiveness levels they belong to overall according to the benchmarking tool Fairea. To answer this RQ, we need to run the code as follows.

(1) We obtain the ML performance and fairness metric values obtained by our approach MAAT (`MAAT/maat.py`). `maat.py` supports three arguments: `-d` configures the dataset; `-c` configures the ML algorithm; `-p` configures the protected attribute.
```
cd MAAT
python maat.py -d adult -c lr -p sex
python maat.py -d adult -c lr -p race
python maat.py -d compas -c lr -p sex
python maat.py -d compas -c lr -p race
python maat.py -d german -c lr -p sex
python maat.py -d bank -c lr -p age
python maat.py -d mep -c rf -p RACE
python maat.py -d adult -c rf -p sex
python maat.py -d adult -c rf -p race
python maat.py -d compas -c rf -p sex
python maat.py -d compas -c rf -p race
python maat.py -d german -c rf -p sex
python maat.py -d bank -c rf -p age
python maat.py -d mep -c rf -p RACE
python maat.py -d adult -c svm -p sex
python maat.py -d adult -c svm -p race
python maat.py -d compas -c svm -p sex
python maat.py -d compas -c svm -p race
python maat.py -d german -c svm -p sex
python maat.py -d bank -c svm -p age
python maat.py -d mep -c svm -p RACE
```
As a result, we can obtain the results of MAAT for 21 (dataset, protected attribute, ML algorithm) combinations. The result for each combination is included in the `RQ1&2_results/` folder. For example, in this folder, `maat_lr_adult_sex.txt` contains the results of MAAT for the (adult, sex, lr) combination. Each file in the folder has 53 columns, with the first column indicating the ML performance or fairness metric, the next 50 columns the metric values of 50 runs, and the last two columns the mean and std values of the 50 runs.

(2) We obtain the ML performance and fairness metric values obtained by existing bias mitigation methods in the ML community: REW (`Fair360/rew.py`), ROC (`Fair360/roc.py`), and ADV (`Fair360/adv.py`). The three methods also support three arguments: `-d` configures the dataset; `-c` configures the ML algorithm; `-p` configures the protected attribute. We take the REW method as an example to show how to run the code.
```
cd Fair360
python rew.py -d adult -c lr -p sex
python rew.py -d adult -c lr -p race
python rew.py -d compas -c lr -p sex
python rew.py -d compas -c lr -p race
python rew.py -d german -c lr -p sex
python rew.py -d bank -c lr -p age
python rew.py -d mep -c rf -p RACE
python rew.py -d adult -c rf -p sex
python rew.py -d adult -c rf -p race
python rew.py -d compas -c rf -p sex
python rew.py -d compas -c rf -p race
python rew.py -d german -c rf -p sex
python rew.py -d bank -c rf -p age
python rew.py -d mep -c rf -p RACE
python rew.py -d adult -c svm -p sex
python rew.py -d adult -c svm -p race
python rew.py -d compas -c svm -p sex
python rew.py -d compas -c svm -p race
python rew.py -d german -c svm -p sex
python rew.py -d bank -c svm -p age
python rew.py -d mep -c svm -p RACE
```
To run the code of ROC and ADV, we just need to replace the `rew.py` as `roc.py` or `adv.py` in the commands above. As a result, we can obtain the results of REW, ROC, and ADV for the 21 combinations of (dataset, protected attribute, ML algorithm). The results for each combination are included in the `RQ1&2_results/` folder.

(3) We obtain the ML performance and fairness metric values obtained by Fairway (`Fairway/`) proposed at ESEC/FSE 2020.  Fairway is implemented for the five datasets (`Adult.py`, `Compas.py`, `German.py`, `Bank.py`, and `MEP.py`), respectively. Each `.py` file supports two arguments: `-c` configures the ML algorithm and `-p` configures the protected attribute.

```
cd Fairway
python Adult.py -c lr -p sex
python Adult.py -c lr -p race
python Compas.py -c lr -p sex
python Compas.py -c lr -p race
python German.py -c lr -p sex
python Bank.py -c lr -p age
python MEP.py -c lr -p RACE
python Adult.py -c rf -p sex
python Adult.py -c rf -p race
python Compas.py -c rf -p sex
python Compas.py -c rf -p race
python German.py -c rf -p sex
python Bank.py -c rf -p age
python MEP.py -c rf -p RACE
python Adult.py -c svm -p sex
python Adult.py -c svm -p race
python Compas.py -c svm -p sex
python Compas.py -c svm -p race
python German.py -c svm -p sex
python Bank.py -c svm -p age
python MEP.py -c svm -p RACE
```
As a result, we can obtain the results of Fairway for 21 (dataset, protected attribute, ML algorithm) combinations. The result for each combination is included in the `RQ1&2_results/` folder.

(4) We obtain the ML performance and fairness metric values obtained by Fair-SMOTE (`Fair-SMOTE/`) proposed at ESEC/FSE 2021. Fair-SMOTE is implemented for the five datasets (`Adult.py`, `Compas.py`, `German.py`, `Bank.py`, and `MEP.py`), respectively. Each `.py` file supports two arguments: `-c` configures the ML algorithm and `-p` configures the protected attribute.

```
cd Fair-SMOTE
python Adult.py -c lr -p sex
python Adult.py -c lr -p race
python Compas.py -c lr -p sex
python Compas.py -c lr -p race
python German.py -c lr -p sex
python Bank.py -c lr -p age
python MEP.py -c lr -p RACE
python Adult.py -c rf -p sex
python Adult.py -c rf -p race
python Compas.py -c rf -p sex
python Compas.py -c rf -p race
python German.py -c rf -p sex
python Bank.py -c rf -p age
python MEP.py -c rf -p RACE
python Adult.py -c svm -p sex
python Adult.py -c svm -p race
python Compas.py -c svm -p sex
python Compas.py -c svm -p race
python German.py -c svm -p sex
python Bank.py -c svm -p age
python MEP.py -c svm -p RACE
```
As a result, we can obtain the results of Fair-SMOTE for 21 (dataset, protected attribute, ML algorithm) combinations. The result for each combination is included in the `RQ1&2_results/` folder.

(5) For each (dataset, protected attribtue, ML algorithm) combination, we use Fairea to construct the fairness-performance trade-off baseline.
```
cd Analysis_code
python cal_baselinepoints.py -d adult -c lr -p sex
python cal_baselinepoints.py -d adult -c lr -p race
python cal_baselinepoints.py -d compas -c lr -p sex
python cal_baselinepoints.py -d compas -c lr -p race
python cal_baselinepoints.py -d german -c lr -p sex
python cal_baselinepoints.py -d bank -c lr -p age
python cal_baselinepoints.py -d mep -c rf -p RACE
python cal_baselinepoints.py -d adult -c rf -p sex
python cal_baselinepoints.py -d adult -c rf -p race
python cal_baselinepoints.py -d compas -c rf -p sex
python cal_baselinepoints.py -d compas -c rf -p race
python cal_baselinepoints.py -d german -c rf -p sex
python cal_baselinepoints.py -d bank -c rf -p age
python cal_baselinepoints.py -d mep -c rf -p RACE
python cal_baselinepoints.py -d adult -c svm -p sex
python cal_baselinepoints.py -d adult -c svm -p race
python cal_baselinepoints.py -d compas -c svm -p sex
python cal_baselinepoints.py -d compas -c svm -p race
python cal_baselinepoints.py -d german -c svm -p sex
python cal_baselinepoints.py -d bank -c svm -p age
python cal_baselinepoints.py -d mep -c svm -p RACE
```
The baselines for each (dataset, protected attribtue, ML algorithm) combination is included in the `Fairea_baseline/` folder. For example, `adult_lr_race_baseline.txt` contains the baseline for the (adult, sex, lr) combination. Each file in the folder has 12 columns, with the first column indicating the ML performance or fairness metric, the second column the metric values of the original model, the next 10 columns the metric values of 10 pseudo models (with different mutation degrees).

(6) We obtain the results for Figure 2.
```
cd Analysis_code
python figure2.py
```

(7) We obtain the results for Table 2. First , we use the `Fair360/default.py` to compute the ML performance and fairness metric values of the original model. To this end, we just need to replace the `rew.py` in the commands in Step(2) with `default.py`. The results for the original model are also included in the `RQ1&2_results/` folder. Then we compare the original model with the models after applying different bias mitigation methods, to obtain the results for Table 2.
```
cd Analysis_code
python table2.py
```

### RQ2 (Applicability): How well does MAAT apply to different ML algorithms, decision tasks, and fairness-performance measurements? 
This RQ analyzes the effectiveness of MAAT on different ML algorithms, decision tasks, and measurements to evaluate its applicability.

We obtain the results for Figure 3 as follows:

```
cd Analysis_code
python figure3.py
```

### RQ3 (Influence of fairness and performance models): How do different fairness models and performance models affect MAAT? 
This RQ adopts different fairness models and performance models to investigate the impacts of the two key components on MAAT.

(1) We first implement M-REW, M-Faiway, and M-Fair-SMOTE to investigate the influence of the fairness model.

* M-REW is implemented as `Fair360/M-rew.py`. To run it, we just need to replace the `rew.py` in the commands in RQ1-Step(2) with `M-rew.py`. The results are included in the `RQ3_results/` folder.

* M-Fairway is implemented as `M-Adult.py`, `M-Compas.py`, `M-German.py`, `M-Bank.py`, and `M-MEP.py` in the `Fairway/` folder. To run them, we just need to replace the `Adult.py`, `Compas.py`, `German.py`, `Bank.py`, and `MEP.py` in the commands in RQ1-Step(3) with `M-Adult.py`, `M-Compas.py`, `M-German.py`, `M-Bank.py`, and `M-MEP.py`. The results are included in the `RQ3_results/` folder.

* M-Fair-SMOTE is implemented as `M-Adult.py`, `M-Compas.py`, `M-German.py`, `M-Bank.py`, and `M-MEP.py` in the `Fair-SMOTE/` folder. To run them, we just need to replace the `Adult.py`, `Compas.py`, `German.py`, `Bank.py`, and `MEP.py` in the commands in RQ1-Step(4) with `M-Adult.py`, `M-Compas.py`, `M-German.py`, `M-Bank.py`, and `M-MEP.py`. The results are included in the `RQ3_results/` folder.

(2) We obtain the results in Table 3 as follows:
```
cd Analysis_code
python table3.py
```

(3) To investigate the influence of the performance model, we take the fairness model trained using the LR algorithm as the example, and change the performance model with the models trained using LR, SVM, RF, and two other popular ML algorithms, i.e., Naive Bayes (NB) and Decision Tree (DT). To obtain the ML performance and fairness metric values obtained by these variants of MAAT, we run as follows:
```
cd MAAT
python maat_change_performance_model.py -d adult -c lr -p sex
python maat_change_performance_model.py -d adult -c rf -p sex
python maat_change_performance_model.py -d adult -c svm -p sex
python maat_change_performance_model.py -d adult -c bayes -p sex
python maat_change_performance_model.py -d adult -c dt -p sex
python maat_change_performance_model.py -d adult -c lr -p race
python maat_change_performance_model.py -d adult -c rf -p race
python maat_change_performance_model.py -d adult -c svm -p race
python maat_change_performance_model.py -d adult -c bayes -p race
python maat_change_performance_model.py -d adult -c dt -p race
python maat_change_performance_model.py -d compas -c lr -p sex
python maat_change_performance_model.py -d compas -c rf -p sex
python maat_change_performance_model.py -d compas -c svm -p sex
python maat_change_performance_model.py -d compas -c bayes -p sex
python maat_change_performance_model.py -d compas -c dt -p sex
python maat_change_performance_model.py -d compas -c lr -p race
python maat_change_performance_model.py -d compas -c rf -p race
python maat_change_performance_model.py -d compas -c svm -p race
python maat_change_performance_model.py -d compas -c bayes -p race
python maat_change_performance_model.py -d compas -c dt -p race
python maat_change_performance_model.py -d bank -c lr -p age
python maat_change_performance_model.py -d bank -c rf -p age
python maat_change_performance_model.py -d bank -c svm -p age
python maat_change_performance_model.py -d bank -c bayes -p age
python maat_change_performance_model.py -d bank -c dt -p age
python maat_change_performance_model.py -d german -c lr -p sex
python maat_change_performance_model.py -d german -c rf -p sex
python maat_change_performance_model.py -d german -c svm -p sex
python maat_change_performance_model.py -d german -c bayes -p sex
python maat_change_performance_model.py -d german -c dt -p sex
python maat_change_performance_model.py -d mep -c lr -p RACE
python maat_change_performance_model.py -d mep -c rf -p RACE
python maat_change_performance_model.py -d mep -c svm -p RACE
python maat_change_performance_model.py -d mep -c bayes -p RACE
python maat_change_performance_model.py -d mep -c dt -p RACE
```
The results are included in the `RQ3_results/` folder.

(4) We obtain the results in Figure 4 as follows:
```
cd Analysis_code
python figure4.py
```

### RQ4 (Influence of combination strategies): How do different combination strategies affect MAAT? 
This RQ investigates different combination strategies for the fairness model and the performance model in MAAT. To this end, we employ 11 strategies, i.e., 0-1, 0.1-0.9, ..., 0.9-0.1, and 1-0, which combine the output probability vectors of the performance model and the fairness model in different proportions. 

(1) We first obtain the ML performance and fairness metric values under different combination strategies.
```
cd MAAT
python maat_combination_strategy.py
```

The results are included in the `RQ4_results/` folder.

(2) We then obtain the results for Figure 5 as follow:
```
cd Analysis_code
python figure5.py
```

### RQ5 (Multiple protected attributes): Is MAAT effective when dealing with multiple protected attributes at the same time?
This RQ investigates whether MAAT provides an effective solution to multi-attribute tasks.

(1) We obtain the ML performance and fairness metrics obtained by MAAT in two multi-attribute tasks (i.e., Adult and Compas).
```
cd MAAT
python maat_multi_attrs.py -d adult -c lr -p1 sex -p2 race
python maat_multi_attrs.py -d adult -c rf -p1 sex -p2 race
python maat_multi_attrs.py -d adult -c svm -p1 sex -p2 race
python maat_multi_attrs.py -d compas -c lr -p1 sex -p2 race
python maat_multi_attrs.py -d compas -c rf -p1 sex -p2 race
python maat_multi_attrs.py -d compas -c svm -p1 sex -p2 race
```
The results are included in the `RQ5_results/` folder.

(2) We obtain the ML performance and fairness metrics obtained by Fair-SMOTE in two multi-attribute tasks (i.e., Adult and Compas).
```
cd Fair-SMOTE
python Adult_Sex_Race.py -c lr
python Adult_Sex_Race.py -c rf
python Adult_Sex_Race.py -c svm
python Compas_Sex_Race.py -c lr
python Compas_Sex_Race.py -c rf
python Compas_Sex_Race.py -c svm
```
The results are included in the `RQ5_results/` folder.

(3) We obtain the results in Figure 6 as follow.
```
cd Analysis_code
python figure6.py
```
## How to Use MAAT in Other Tasks
It is easy to use MAAT for other decision-making tasks. You just need to revise the `MAAT/utility.py` as follows:

(1) The `get_data()` function in the `MAAT/utility.py` specifies the dataset used for training the decision-making models. You can configure the datasets that they would like to use in this function.

(2) The `get_classifier()` function in the `MAAT/utility.py` specifies the machine learning algorithm used for training models. You can configure the algorithm that they would like to use in this function.


## Declaration
Thanks to the authors of existing bias mitigation methods for open source, to facilitate our implementation of this paper. Therefore, when using our code or data for your work, please also consider citing their papers, including [AIF360](https://arxiv.org/abs/1810.01943), [Fairway](https://doi.org/10.1145/3368089.3409697), [Fair-SMOTE](https://doi.org/10.1145/3468264.3468537), and [Fairea](https://doi.org/10.1145/3468264.3468565).

## Citation
Please consider citing the following paper when using our code or data.
```
@inproceedings{zhenpengmaat2022,
  title={MAAT: A Novel Ensemble Approach to Addressing Fairness and Performance Bugs for Machine Learning Software},
  author={Zhenpeng Chen and Jie M. Zhang and Federica Sarro and Mark Harman},
  booktitle={Proceedings of the 2022 ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE'22},
  year={2022}
}
```

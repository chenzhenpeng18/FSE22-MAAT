# [ESEC/FSE 2022] MAAT: A Novel Ensemble Approach to Addressing Fairness and Performance Bugs for Machine Learning Software

Welcome to visit the homepage of our ESEC/FSE'22 paper entitled "MAAT: A Novel Ensemble Approach to Addressing Fairness and Performance Bugs for Machine Learning Software". The homepage contains scripts and data used in this paper.

## Experimental environment

We use Python 3.7.11 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit for implementing bias mitigation methods and computing fairness metrics. This toolkit can be installed as follows:

```
pip install aif360
```

More information on AIF360 can be found on https://github.com/Trusted-AI/AIF360.

In addition, we require the following Python packages:
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
```

## Dataset

We use the five default datasets supported by the AIF360 toolkit. Please refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files.

## Scripts and results
The repository contains five folders:

* ```MAAT/``` contains code for implementing our approach.
  
* ```Fair360/``` contains code for implementing three bias mitigation methods (i.e., REW, ADV, and ROC) in AIF360.

* ```Fairway/``` contains code for implementing Fairway, a bias mitigation method proposed by [Chakraborty et al.](https://doi.org/10.1145/3368089.3409697) in ESEC/FSE 2020.

* ```Fair-SMOTE/``` contains code for implementing Fair-SMOTE, a bias mitigation method proposed by [Chakraborty et al.](https://doi.org/10.1145/3468264.3468537) in ESEC/FSE 2021.

* ```Fairea/``` contains code for benchmarking fairness-performance trade-off. It is implemented based on Fairea, a trade-off benchmark method proposed by [Hort et al.](https://doi.org/10.1145/3468264.3468565) in ESEC/FSE 2021.

* ```Analysis_code/``` contains code for calculating the baselines for Fairea and the code for analyzing the effectiveness distribution.
  
* ```Fairea_baseline/``` contains the performance and fairness metric values of the data points that construct the Fairea baselines.

* ```RQ1&2_results/```, ```RQ3_results/```, ```RQ4_results/```, and ```RQ5_results/``` contain the raw results of the models after applying different bias mitigation methods/strategies. Each file in these folders has 53 columns, with the first column indicating the metric, the next 50 columns the metric values of 50 runs, and the last two columns the mean and std values of the 50 runs.

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

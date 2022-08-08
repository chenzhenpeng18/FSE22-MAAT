# Installation instructions

This document contains information on how to install packages and datasets.

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
pip install aif360
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
```

## Dataset

We use the five default datasets supported by the AIF360 toolkit. When running the scripts that invoke these datasets, you will be prompted how to download these datasets and in which folders they need to be placed. You can also refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files.

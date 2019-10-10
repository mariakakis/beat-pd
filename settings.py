import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import os

# Training parameters
MIN_OBSERVATIONS_PER_SUBJECT = 40
MIN_OBSERVATIONS_PER_CLASS = 10
NUM_STRATIFIED_FOLDS = 5
NUM_STRATIFIED_ROUNDS = 1
PARAM_SEARCH_FOLDS = 5

if os.name == 'nt':
    HOME_DIRECTORY = 'C:\\Users\\atm15.CSENETID\\Desktop\\beat-pd'
else:
    HOME_DIRECTORY = '~\\Desktop\\beat-pd'

# Classifiers
CLASSIF_RANDOM_FOREST = 'classif-rf'
CLASSIF_XGBOOST = 'classif-xg'
CLASSIF_ORDINAL_LOGISTIC = 'classif-ord-log'
CLASSIF_ORDINAL_RANDOM_FOREST = 'classif-ord-rf'
CLASSIF_MLP = 'classif-mlp'
CLASSIFIERS = [CLASSIF_RANDOM_FOREST, CLASSIF_XGBOOST, CLASSIF_MLP,
               CLASSIF_ORDINAL_RANDOM_FOREST, CLASSIF_ORDINAL_LOGISTIC]

# Regressors
REGRESS_XGBOOST = 'regress-xg'
REGRESS_MLP = 'regress-mlp'
REGRESSORS = [REGRESS_XGBOOST, REGRESS_MLP]

RUN_PARALLEL = False
NUM_THREADS = 3
DEBUG = True
RANDOM_SEED = 812

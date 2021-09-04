import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import os
import csv
import synapseclient
import pickle

DEBUG = False
NUM_THREADS = 2
RANDOM_SEED = 812

# Training parameters
NUM_STRATIFIED_FOLDS = 10 if not DEBUG else 2
FRAC_VALIDATION_DATA = 0.2
PARAM_SEARCH_FOLDS = 3

if os.name == 'nt':
    HOME_DIRECTORY = os.path.join('C:\\', 'Users', 'atm15.CSENETID', 'Desktop', 'beat-pd')
    RUN_PARALLEL = True if not DEBUG else False
else:
    HOME_DIRECTORY = os.path.join('/Users', 'alex', 'Desktop', 'beat-pd')
    RUN_PARALLEL = False

# Classifiers
CLASSIF_RANDOM_FOREST = 'classif-rf'
CLASSIF_XGBOOST = 'classif-xg'
CLASSIF_MLP = 'classif-mlp'
CLASSIF_ORDINAL_LOGISTIC = 'classif-ord-log'
CLASSIF_ORDINAL_RANDOM_FOREST = 'classif-ord-rf'
CLASSIFIERS = []

# Regressors
REGRESS_XGBOOST = 'regress-xg'
REGRESS_MLP = 'regress-mlp'
REGRESSORS = [REGRESS_XGBOOST, REGRESS_MLP]

# Run parameters
DATASET_CIS, DATASET_REAL = 1, 2
FEATURE_SOURCE_NICK, FEATURE_SOURCE_PHIL = 1, 2
SPLIT_STRUCTURE_RANDOM, SPLIT_STRUCTURE_DEFINED = 1, 2
SENSOR_WATCH_ACCEL, SENSOR_WATCH_GYRO, SENSOR_PHONE_ACCEL, SENSOR_ALL = 1, 2, 3, 4

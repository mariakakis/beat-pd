import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import os

NUM_STRATIFIED_FOLDS = 5
NUM_STRATIFIED_ROUNDS = 1
PARAM_SEARCH_FOLDS = 5

MIN_OBSERVATIONS_PER_SUBJECT = 40
MIN_OBSERVATIONS_PER_CLASS = 10 #2

RANDOM_FOREST = 'rf'
XGBOOST = 'xg'
ORDINAL = 'ordinal'
ORDINAL_RANDOM_FOREST = 'ord-rf'

RUN_PARALLEL = False

DEBUG = False

RANDOM_SEED = 812

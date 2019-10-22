from settings import *
from model_training.classif_trainer import train_user_classification
from model_training.regress_trainer import train_user_regression
from joblib import Parallel, delayed
import itertools
from model_training.helpers import make_dir

# Get run params
cis_or_real = int(input('Dataset: (1) CIS or (2) REAL:'))
nick_or_sage = int(input('Feature source: (1) Nick or (2) Sage:'))
data_source = int(input('Sensor features: (1) Watch accel, (2) Watch gyro, '
                        '(3) Phone accel, (4) Phone gyro:'))

# Read data files using run params
Data, Meta = None, None
# CIS PD
if cis_or_real == 1:
    Meta = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data', 'CIS', 'Metadata.csv'))
    if nick_or_sage == 1 and data_source == 1:
        Data = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data', 'CIS', 'WatchAcc_Nick_Features.csv'))
# REAL PD
elif cis_or_real == 2:
    Meta = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data', 'REAL', 'Metadata.csv'))
    if nick_or_sage == 1:
        if data_source == 1:
            Data = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data', 'REAL', 'WatchAcc_Nick_Features.csv'))
        elif data_source == 2:
            Data = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data', 'REAL', 'WatchGyro_Nick_Features.csv'))
        elif data_source == 3:
            Data = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data', 'REAL', 'PhoneAcc_Nick_Features.csv'))
    elif nick_or_sage == 2 and data_source == 1:

        Data = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data', 'REAL', 'WatchAcc_Sage_Features.tsv'), sep='\t')
if Data is None or Meta is None:
    raise ValueError('Not a valid dataset input')

# Get all of the metadata into the main data frame
Data = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data/CIS/WatchAcc_Nick_Features.csv'))
Meta = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data/CIS/Metadata.csv'))
Meta.set_index('measurement_id', inplace=True)
Data['subject_id'] = Data.ID.apply(lambda x: Meta.loc[x, 'subject_id'])
Data['timestamp'] = Data.ID.apply(lambda x: Meta.loc[x, 'timestamp'])
Data['activity_intensity'] = Data.ID.apply(lambda x: Meta.loc[x, 'activity_intensity'])
Data['dyskinesia'] = Data.ID.apply(lambda x: Meta.loc[x, 'dyskinesia'])
Data['on_off'] = Data.ID.apply(lambda x: Meta.loc[x, 'on_off'])
Data['tremor'] = Data.ID.apply(lambda x: Meta.loc[x, 'tremor'])
print('Done processing data')

# Make directories
make_dir(HOME_DIRECTORY)
make_dir(os.path.join(HOME_DIRECTORY, 'output'))
make_dir(os.path.join(HOME_DIRECTORY, 'output', 'model_training'))
make_dir(os.path.join(HOME_DIRECTORY, 'output', 'regression'))

# Train classifier for each label
label_names = ['on_off', 'dyskinesia', 'tremor']
if not RUN_PARALLEL:
    for label_name in label_names:
        for model_type in CLASSIFIERS:
            train_user_classification(Data, label_name, model_type)
else:
    combinations = list(itertools.product(label_names, CLASSIFIERS))
    Parallel(n_jobs=NUM_THREADS)(delayed(train_user_classification)(Data, label_name, model_type)
                                 for (label_name, model_type) in combinations)

# Train regressor for each label
if not RUN_PARALLEL:
    for label_name in label_names:
        for model_type in REGRESSORS:
            train_user_regression(Data, label_name, model_type)
else:
    combinations = list(itertools.product(label_names, REGRESSORS))
    Parallel(n_jobs=NUM_THREADS)(delayed(train_user_regression)(Data, label_name, model_type)
                                 for (label_name, model_type) in combinations)

from settings import *
from model_training.classif_trainer import train_user_classification
from model_training.regress_trainer import train_user_regression
from joblib import Parallel, delayed
import itertools
from model_training.helpers import make_dir
from model_training.view_distribution import view_distribution


def run_training(syn):
    # Get run params
    cis_or_real = int(input('Dataset: (1) CIS or (2) REAL: '))
    nick_or_sage = int(input('Feature source: (1) Nick or (2) Sage: '))
    data_source = int(input('Sensor features: (1) Watch accel, (2) Watch gyro, '
                            '(3) Phone accel: '))

    # Read data files using run params
    data, metadata = None, None
    # CIS PD
    if cis_or_real == 1:
        metadata = syn.tableQuery("select * from syn20489608").asDataFrame()
        if nick_or_sage == 1 and data_source == 1:
            data = pd.read_csv(syn.get('syn20712268').path)
    # REAL PD
    elif cis_or_real == 2:
        metadata = syn.tableQuery("select * from syn20822276").asDataFrame()
        if nick_or_sage == 1:
            if data_source == 1:
                data = pd.read_csv(syn.get('syn20928640').path)
            elif data_source == 2:
                data = pd.read_csv(syn.get('syn20928636').path)
            elif data_source == 3:
                data = pd.read_csv(syn.get('syn20928641').path)
        elif nick_or_sage == 2 and data_source == 1:
            data = pd.read_csv(syn.get('syn21042208').path, sep='\t')
    if data is None or metadata is None:
        raise ValueError('Not a valid dataset input')
    print('Valid run params')

    # Handle specific data formats
    if cis_or_real == 2 and nick_or_sage == 1 and (data_source == 1 or data_source == 2):
        data = data.drop('Unnamed: 0', axis=1)
    if nick_or_sage == 2:
        data = data.drop(['sensor_location', 'sensor', 'measurementType', 'axis', 'window',
                          'window_start_time', 'window_end_time'], axis=1)
        data = data.rename(columns={'measurement_id': 'ID'})
        col_names = data.columns.tolist()
        col_names = col_names[1:] + col_names[:1]
        data = data[col_names]

    # Handle specific metadata formats
    timestamp_colname = 'timestamp' if cis_or_real == 1 else 'reported_timestamp'

    # Get all of the metadata into the main data frame
    metadata.set_index('measurement_id', inplace=True)
    data['subject_id'] = data.ID.apply(lambda x: metadata.loc[x, 'subject_id'])
    data['timestamp'] = data.ID.apply(lambda x: metadata.loc[x, timestamp_colname])
    data['dyskinesia'] = data.ID.apply(lambda x: metadata.loc[x, 'dyskinesia'])
    data['on_off'] = data.ID.apply(lambda x: metadata.loc[x, 'on_off'])
    data['tremor'] = data.ID.apply(lambda x: metadata.loc[x, 'tremor'])
    print('Done processing data')

    # View distribution
    # view_distribution(data)

    # Make directories
    make_dir(HOME_DIRECTORY)
    make_dir(os.path.join(HOME_DIRECTORY, 'output'))
    make_dir(os.path.join(HOME_DIRECTORY, 'output', 'classification'))
    make_dir(os.path.join(HOME_DIRECTORY, 'output', 'regression'))

    # Train classifier for each label
    label_names = ['on_off', 'dyskinesia', 'tremor']
    if not RUN_PARALLEL:
        for label_name in label_names:
            for model_type in CLASSIFIERS:
                train_user_classification(data, label_name, model_type)
    else:
        combinations = list(itertools.product(label_names, CLASSIFIERS))
        Parallel(n_jobs=NUM_THREADS)(delayed(train_user_classification)(data, label_name, model_type)
                                     for (label_name, model_type) in combinations)

    # Train regressor for each label
    if not RUN_PARALLEL:
        for label_name in label_names:
            for model_type in REGRESSORS:
                train_user_regression(data, label_name, model_type)
    else:
        combinations = list(itertools.product(label_names, REGRESSORS))
        Parallel(n_jobs=NUM_THREADS)(delayed(train_user_regression)(data, label_name, model_type)
                                     for (label_name, model_type) in combinations)

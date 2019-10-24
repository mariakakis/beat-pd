from settings import *
from model_training.classif_trainer import train_user_classification
from model_training.regress_trainer import train_user_regression
from joblib import Parallel, delayed
import itertools
from model_training.helpers import make_dir
from synapseclient import File
from model_training.view_distribution import view_distribution

# Login to synapse
syn = synapseclient.Synapse()
syn.login()

# Create directories
run_id = input('Run id: ')
run_folder = os.path.join(HOME_DIRECTORY, 'output', run_id)
make_dir(HOME_DIRECTORY)
make_dir(os.path.join(HOME_DIRECTORY, 'output'))
run_settings_file = os.path.join(run_folder, 'settings.pkl')

# Either load settings or ask for them
if os.path.exists(run_settings_file):
    settings = pickle.load(open(run_settings_file, 'rb'))
    cis_or_real = settings['cis_or_real']
    feature_source = settings['feature_source']
    split_structure = settings['split_structure']
    data_source = settings['data_source']
else:
    cis_or_real = int(input('Dataset: (1) CIS or (2) REAL: '))
    feature_source = int(input('Feature source: (1) Nick or (2) Phil: '))
    split_structure = int(input('Split structure source: (1) random or (2) pre-defined: '))
    data_source = int(input('Sensor features: (1) Watch accel, (2) Watch gyro, '
                            '(3) Phone accel: '))
    make_dir(run_folder)
    output = open(run_settings_file, 'wb')
    pickle.dump({'cis_or_real': cis_or_real, 'feature_source': feature_source,
                 'split_structure': split_structure, 'data_source': data_source}, output)
    output.close()

# Read data files using run params
data, metadata, splits = None, None, None
if cis_or_real == DATASET_CIS:
    # Metadata
    if split_structure == SPLIT_STRUCTURE_RANDOM:
        metadata = syn.tableQuery("select * from syn20489608").asDataFrame()
    elif split_structure == SPLIT_STRUCTURE_DEFINED:
        metadata = pd.read_csv(syn.get('syn21036470').path)

    # Data
    if feature_source == FEATURE_SOURCE_NICK and data_source == SENSOR_WATCH_ACCEL:
        data = pd.read_csv(syn.get('syn20712268').path)
elif cis_or_real == DATASET_REAL:
    # Metadata
    metadata = syn.tableQuery("select * from syn20822276").asDataFrame()

    # Data
    if feature_source == FEATURE_SOURCE_NICK and data_source == SENSOR_WATCH_ACCEL:
        data = pd.read_csv(syn.get('syn20928640').path)
    elif feature_source == FEATURE_SOURCE_NICK and data_source == SENSOR_WATCH_GYRO:
        data = pd.read_csv(syn.get('syn20928636').path)
    elif feature_source == FEATURE_SOURCE_NICK and data_source == SENSOR_PHONE_ACCEL:
        data = pd.read_csv(syn.get('syn20928641').path)
    elif feature_source == FEATURE_SOURCE_PHIL and data_source == SENSOR_WATCH_ACCEL:
        data = pd.read_csv(syn.get('syn21042208').path, sep='\t')
if data is None or metadata is None:
    raise ValueError('Not a valid dataset input')
print('Valid run params')

# Handle specific data formats
if cis_or_real == DATASET_REAL and feature_source == FEATURE_SOURCE_NICK and \
        (data_source == SENSOR_WATCH_ACCEL or data_source == SENSOR_WATCH_GYRO):
    data = data.drop('Unnamed: 0', axis=1)
if feature_source == FEATURE_SOURCE_PHIL:
    data = data.drop(['sensor_location', 'sensor', 'measurementType', 'axis', 'window',
                      'window_start_time', 'window_end_time'], axis=1)
    data = data.rename(columns={'measurement_id': 'ID'})
    col_names = data.columns.tolist()
    col_names = col_names[1:] + col_names[:1]
    data = data[col_names]

# Remove cases when measurement_id is in data but not meta
metadata.set_index('measurement_id', inplace=True)
data = data[data['ID'].isin(metadata.index)]

# Get all of the metadata into the main data frame
data['subject_id'] = data.ID.apply(lambda x: metadata.loc[x, 'subject_id'])
data['dyskinesia'] = data.ID.apply(lambda x: metadata.loc[x, 'dyskinesia'])
data['on_off'] = data.ID.apply(lambda x: metadata.loc[x, 'on_off'])
data['tremor'] = data.ID.apply(lambda x: metadata.loc[x, 'tremor'])
for fold_idx in range(NUM_STRATIFIED_FOLDS):
    if split_structure == SPLIT_STRUCTURE_DEFINED:
        data['fold_%d' % fold_idx] = data.ID.apply(lambda x: metadata.loc[x, 'training%d' % (fold_idx+1)])
    elif split_structure == SPLIT_STRUCTURE_RANDOM:
        data['fold_%d' % fold_idx] = np.nan
print('Done processing data')

# View distribution
# view_distribution(data)

# Train model for each label
label_names = ['on_off', 'dyskinesia', 'tremor']
csv_files, img_files = [], []
if not RUN_PARALLEL:
    for label_name in label_names:
        for model_type in CLASSIFIERS:
            csv_file, img_file = train_user_classification(data, label_name, model_type, run_id)
            csv_files.append(csv_file)
            img_files.append(img_file)
        for model_type in REGRESSORS:
            csv_file, img_file = train_user_regression(data, label_name, model_type, run_id)
            csv_files.append(csv_file)
            img_files.append(img_file)
else:
    combinations = list(itertools.product(label_names, CLASSIFIERS))
    results = Parallel(n_jobs=NUM_THREADS)(delayed(train_user_classification)(data, label_name, model_type, run_id)
                                           for (label_name, model_type) in combinations)
    for i in range(len(combinations)):
        csv_files.append(results[i][0])
        img_files.append(results[i][1])

    combinations = list(itertools.product(label_names, REGRESSORS))
    results = Parallel(n_jobs=NUM_THREADS)(delayed(train_user_regression)(data, label_name, model_type, run_id)
                                           for (label_name, model_type) in combinations)
    for i in range(len(combinations)):
        csv_files.append(results[i][0])
        img_files.append(results[i][1])

# TODO: zip results

# TODO: upload to synapse with old or new synapse ID?
# new_file = syn.store(File('path/to/new_version/raw_data.txt', parentId='syn123456'))

# Logout of synapse
syn.logout()

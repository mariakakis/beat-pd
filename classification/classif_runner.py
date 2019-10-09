from settings import *
from classification.classif_trainer import train_user_model
from joblib import Parallel, delayed
import itertools


# Get all of the metadata into the main data frame
Data = pd.read_csv('../Watch_Features_Full.csv')
Meta = pd.read_csv('../Metadata.csv')
Meta.set_index('measurement_id', inplace=True)
Data['subject_id'] = Data.ID.apply(lambda x: Meta.loc[x, 'subject_id'])
Data['timestamp'] = Data.ID.apply(lambda x: Meta.loc[x, 'timestamp'])
Data['activity_intensity'] = Data.ID.apply(lambda x: Meta.loc[x, 'activity_intensity'])
Data['dyskinesia'] = Data.ID.apply(lambda x: Meta.loc[x, 'dyskinesia'])
Data['on_off'] = Data.ID.apply(lambda x: Meta.loc[x, 'on_off'])
Data['tremor'] = Data.ID.apply(lambda x: Meta.loc[x, 'tremor'])
print('Done processing data')

# Train for each label
label_names = ['on_off', 'dyskinesia', 'tremor']
if not RUN_PARALLEL:
    for model_type in CLASSIFIERS:
        for label_name in label_names:
            train_user_model(Data, label_name, model_type)
else:
    combinations = list(itertools.product(label_names, CLASSIFIERS))
    Parallel(n_jobs=2)(delayed(train_user_model)(Data, label_name, model_type)
                       for (label_name, model_type) in combinations)

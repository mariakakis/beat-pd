import pandas as pd
from settings import *
from training_helper import train_user_model

# Get all of the metadata into the main data frame
Data = pd.read_csv('./Watch_Features_Full.csv')
Meta = pd.read_csv('./Metadata.csv')
Meta.set_index('measurement_id', inplace=True)
Data['subject_id'] = Data.ID.apply(lambda x: Meta.loc[x, 'subject_id'])
Data['timestamp'] = Data.ID.apply(lambda x: Meta.loc[x, 'timestamp'])
Data['activity_intensity'] = Data.ID.apply(lambda x: Meta.loc[x, 'activity_intensity'])
Data['dyskinesia'] = Data.ID.apply(lambda x: Meta.loc[x, 'dyskinesia'])
Data['on_off'] = Data.ID.apply(lambda x: Meta.loc[x, 'on_off'])
Data['tremor'] = Data.ID.apply(lambda x: Meta.loc[x, 'tremor'])
print('Done processing data')

# Train for each label
for model_type in (RANDOM_FOREST, XGBOOST, ORDINAL):
    for label_type in ('on_off', 'dyskinesia', 'tremor'):
        print('Model:', model_type, ', Label:', label_type)
        train_user_model(Data, label_type, model_type)
        print('**********************')

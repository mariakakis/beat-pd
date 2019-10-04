import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from settings import *

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

n_classes = 5
sorted_subjects = sorted(Data.subject_id.unique())
for label_name in ['on_off', 'dyskinesia', 'tremor']:
    cnts = [list() for _ in range(n_classes)]
    for subject in sorted_subjects:
        subj_data = Data[Data.subject_id == subject].copy()
        subj_data = subj_data[subj_data.on_off > -1]
        subj_data = subj_data.set_index(["subject_id"])

        user_cnts = subj_data[label_name].value_counts()
        for i in range(n_classes):
            cnts[i].append(user_cnts[i] if i in user_cnts else 0)

    cnts = np.array(cnts)
    norm_cnts = cnts/cnts.sum(axis=0)[None, :]
    ind = range(len(sorted_subjects))

    fig = plt.figure(figsize=(13, 13))
    sns.set(style="whitegrid")
    ax = fig.add_subplot(211)
    plt.title(label_name)
    p0 = plt.bar(ind, cnts[0])
    p1 = plt.bar(ind, cnts[1], bottom=cnts[0])
    p2 = plt.bar(ind, cnts[2], bottom=cnts[0]+cnts[1])
    p3 = plt.bar(ind, cnts[3], bottom=cnts[0]+cnts[1]+cnts[2])
    p4 = plt.bar(ind, cnts[4], bottom=cnts[0]+cnts[1]+cnts[2]+cnts[3])
    plt.xticks(ind, sorted_subjects), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID'), plt.ylabel('Count')
    plt.legend((p0[0], p1[0], p2[0], p3[0], p4[0]), ('0', '1', '2', '3', '4'))

    ax = fig.add_subplot(212)
    p0 = plt.bar(ind, norm_cnts[0])
    p1 = plt.bar(ind, norm_cnts[1], bottom=norm_cnts[0])
    p2 = plt.bar(ind, norm_cnts[2], bottom=norm_cnts[0]+norm_cnts[1])
    p3 = plt.bar(ind, norm_cnts[3], bottom=norm_cnts[0]+norm_cnts[1]+norm_cnts[2])
    p4 = plt.bar(ind, norm_cnts[4], bottom=norm_cnts[0]+norm_cnts[1]+norm_cnts[2]+norm_cnts[3])
    plt.xticks(ind, sorted_subjects), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID'), plt.ylabel('Normalized Count')

    plt.show()

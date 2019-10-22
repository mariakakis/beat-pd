from settings import *

# Get all of the metadata into the main data frame
Data = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data/WatchAcc_Nick_Features.csv'))
Meta = pd.read_csv(os.path.join(HOME_DIRECTORY, 'data/Metadata.csv'))
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
    pre_counts = [list() for _ in range(n_classes)]
    post_counts = [list() for _ in range(n_classes)]
    for subject in sorted_subjects:
        subj_data = Data[Data.subject_id == subject].copy()
        subj_data = subj_data[subj_data.on_off > -1]
        subj_data = subj_data.set_index(["subject_id"])

        # Make a table that just has unique measurement_ids and labels for the user
        id_table = subj_data[['ID', label_name]].drop_duplicates()

        # Update the pre_counts
        label_counts = id_table[label_name].value_counts()
        for i in range(n_classes):
            if i in label_counts:
                pre_counts[i].append(label_counts[i])
            else:
                pre_counts[i].append(0)

        # Remove any classes with not enough samples
        for i in range(len(label_counts)):
            if i in label_counts and label_counts[i] <= MIN_OBSERVATIONS_PER_CLASS:
                subj_data = subj_data[subj_data[label_name] != i]
                id_table = id_table[id_table[label_name] != i]

        # Update post_counts
        label_counts = id_table[label_name].value_counts()
        for i in range(n_classes):
            if i in label_counts:
                post_counts[i].append(label_counts[i] if
                                      len(id_table) > MIN_OBSERVATIONS_PER_SUBJECT else 0)
            else:
                post_counts[i].append(0)

    pre_counts = np.array(pre_counts)
    norm_pre_counts = pre_counts/pre_counts.sum(axis=0)[None, :]
    norm_pre_counts = np.nan_to_num(norm_pre_counts)
    post_counts = np.array(post_counts)
    norm_post_counts = post_counts/post_counts.sum(axis=0)[None, :]
    norm_post_counts = np.nan_to_num(norm_post_counts)
    ind = range(len(sorted_subjects))

    fig = plt.figure(figsize=(13, 13))
    sns.set(style="whitegrid")
    ax = fig.add_subplot(221)
    plt.title('%s: Pre-filtering' % label_name)
    p10 = plt.bar(ind, pre_counts[0])
    p11 = plt.bar(ind, pre_counts[1], bottom=pre_counts[0])
    p12 = plt.bar(ind, pre_counts[2], bottom=pre_counts[0]+pre_counts[1])
    p13 = plt.bar(ind, pre_counts[3], bottom=pre_counts[0]+pre_counts[1]+pre_counts[2])
    p14 = plt.bar(ind, pre_counts[4], bottom=pre_counts[0]+pre_counts[1]+pre_counts[2]+pre_counts[3])
    plt.xticks(ind, sorted_subjects), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID'), plt.ylabel('Count Pre Filtering')
    plt.legend((p10[0], p11[0], p12[0], p13[0], p14[0]), ('0', '1', '2', '3', '4'))

    ax = fig.add_subplot(223)
    p30 = plt.bar(ind, norm_pre_counts[0])
    p31 = plt.bar(ind, norm_pre_counts[1], bottom=norm_pre_counts[0])
    p32 = plt.bar(ind, norm_pre_counts[2], bottom=norm_pre_counts[0]+norm_pre_counts[1])
    p33 = plt.bar(ind, norm_pre_counts[3], bottom=norm_pre_counts[0]+norm_pre_counts[1]+norm_pre_counts[2])
    p34 = plt.bar(ind, norm_pre_counts[4], bottom=norm_pre_counts[0]+norm_pre_counts[1]+norm_pre_counts[2]+norm_pre_counts[3])
    plt.xticks(ind, sorted_subjects), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID'), plt.ylabel('Normalized Count Pre Filtering')

    ax = fig.add_subplot(222)
    plt.title('%s: Post-filtering' % label_name)
    p20 = plt.bar(ind, post_counts[0])
    p21 = plt.bar(ind, post_counts[1], bottom=post_counts[0])
    p22 = plt.bar(ind, post_counts[2], bottom=post_counts[0]+post_counts[1])
    p23 = plt.bar(ind, post_counts[3], bottom=post_counts[0]+post_counts[1]+post_counts[2])
    p24 = plt.bar(ind, post_counts[4], bottom=post_counts[0]+post_counts[1]+post_counts[2]+post_counts[3])
    plt.xticks(ind, sorted_subjects), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID'), plt.ylabel('Count Pre Filtering')
    plt.legend((p20[0], p21[0], p22[0], p23[0], p24[0]), ('0', '1', '2', '3', '4'))

    ax = fig.add_subplot(224)
    p40 = plt.bar(ind, norm_post_counts[0])
    p41 = plt.bar(ind, norm_post_counts[1], bottom=norm_post_counts[0])
    p42 = plt.bar(ind, norm_post_counts[2], bottom=norm_post_counts[0]+norm_post_counts[1])
    p43 = plt.bar(ind, norm_post_counts[3], bottom=norm_post_counts[0]+norm_post_counts[1]+norm_post_counts[2])
    p44 = plt.bar(ind, norm_post_counts[4], bottom=norm_post_counts[0]+norm_post_counts[1]+norm_post_counts[2]+norm_post_counts[3])
    plt.xticks(ind, sorted_subjects), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID'), plt.ylabel('Normalized Count Pre Filtering')

    plt.savefig(os.path.join('output', 'distributions', '%s.png' % label_name), bbox_inches='tight')
    plt.show()

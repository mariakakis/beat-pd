from settings import *
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import label_binarize
from scipy import stats
import errno


def combine_data(watch_accel, watch_gyro, phone_accel):
    # Rename columns
    watch_accel.rename(columns=lambda x: '%s_watchaccel' % x if x != 'ID' else x, inplace=True)
    watch_gyro.rename(columns=lambda x: '%s_watchgyro' % x if x != 'ID' else x, inplace=True)
    phone_accel.rename(columns=lambda x: '%s_phoneaccel' % x if x != 'ID' else x, inplace=True)

    # Join based on measurement id
    data = pd.merge(watch_accel, watch_gyro, on='ID', how='left')
    data = pd.merge(data, phone_accel, on='ID', how='left')
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def preprocess_data(id_table, subject, label_name):
    # Get data belonging to a specific subject
    subj_id_table = id_table[id_table.subject_id == subject].copy()

    # Remove cases without a label
    subj_id_table = subj_id_table[~np.isnan(subj_id_table[label_name])]
    subj_id_table = subj_id_table[subj_id_table[label_name] >= 0]

    # Remove any classes with not enough samples
    label_counts = subj_id_table[label_name].value_counts()
    for i in range(len(label_counts)):
        if i in label_counts and label_counts[i] <= MIN_OBSERVATIONS_PER_CLASS:
            subj_id_table = subj_id_table[subj_id_table[label_name] != i]
            print_debug('Removing class %d from this user' % i)

    # Skip if not enough data left over
    if len(subj_id_table) <= MIN_OBSERVATIONS_PER_SUBJECT:
        print_debug('Not enough data points for that subject')
        return None, None

    # Create folds
    if any([col.startswith('fold') for col in subj_id_table.columns.tolist()]):
        folds = []
        num_folds = len(list(filter(lambda x: x.startswith('fold'), subj_id_table.columns.tolist())))
        for fold_idx in range(num_folds):
            train_idxs = np.where(subj_id_table['fold_%d' % fold_idx])[0]
            test_idxs = np.where(~subj_id_table['fold_%d' % fold_idx])[0]
            folds.append((train_idxs, test_idxs))
    else:
        skf = StratifiedKFold(n_splits=NUM_STRATIFIED_FOLDS, random_state=RANDOM_SEED)
        folds = list(skf.split(subj_id_table.ID, subj_id_table[label_name]))

    return subj_id_table, folds


def calculate_scores(y_train, y_test, train_classes, test_classes, subj_data_test, preds, probs):
    # Bin probabilities over each diary entry
    y_test_bin, preds_bin, probs_bin = [], [], []
    test_data_ids = subj_data_test['ID']
    for ID in np.unique(test_data_ids):
        y_test_bin.append(np.mean(y_test[test_data_ids == ID]))
        preds_bin.append(np.mean(preds[test_data_ids == ID]))
        probs_bin.append(np.mean(probs[test_data_ids == ID, :], axis=0).reshape([1, -1]))
    y_test_bin = np.vstack(y_test_bin)
    preds_bin = np.vstack(preds_bin)
    probs_bin = np.vstack(probs_bin)

    # Binarize the results
    y_test_binary = label_binarize(y_test, train_classes)
    y_test_bin_binary = label_binarize(y_test_bin, train_classes)

    # Drop probabilities for classes not found in test data
    for i in list(range(np.shape(y_test_binary)[1]))[::-1]:
        if not any(y_test_bin_binary[:, i]):
            y_test_binary = np.delete(y_test_binary, i, axis=1)
            y_test_bin_binary = np.delete(y_test_bin_binary, i, axis=1)
            probs = np.delete(probs, i, axis=1)
            probs_bin = np.delete(probs_bin, i, axis=1)

    # Calculate MSE/MAE
    mse = mean_squared_error(y_test_bin, preds_bin)
    vse = var_squared_error(y_test_bin, preds_bin)
    mae = mean_absolute_error(y_test_bin, preds_bin)
    vae = var_absolute_error(y_test_bin, preds_bin)

    # Compute null model MSE/MAE and the gain
    mse_trivial = np.ones(preds_bin.shape) * np.mean(y_train)
    mae_trivial = np.ones(preds_bin.shape) * np.median(y_train)
    null_mse = mean_squared_error(y_test_bin, mse_trivial)
    null_vse = var_squared_error(y_test_bin, mse_trivial)
    null_mae = mean_absolute_error(y_test_bin, mae_trivial)
    null_vae = var_absolute_error(y_test_bin, mse_trivial)

    # Compute macro-MSE/MAE
    macro_mse, macro_mae = 0, 0
    macro_vse, macro_vae = 0, 0
    for c in test_classes:
        idxs = np.where(y_test_bin == c)
        macro_mse += mean_squared_error(y_test_bin[idxs], preds_bin[idxs]) / len(train_classes)
        macro_vse += var_squared_error(y_test_bin[idxs], preds_bin[idxs]) / len(train_classes)
        macro_mae += mean_absolute_error(y_test_bin[idxs], preds_bin[idxs]) / len(train_classes)
        macro_vae += var_absolute_error(y_test_bin[idxs], preds_bin[idxs]) / len(train_classes)

    # Compute null model macro-MSE/macro-MAE and the gain
    null_macro_mse, null_macro_mae = 0, 0
    null_macro_vse, null_macro_vae = 0, 0
    macro_mse_trivial = np.ones(preds_bin.shape) * np.mean(train_classes)
    macro_mae_trivial = np.ones(preds_bin.shape) * np.median(train_classes)
    for c in test_classes:
        idxs = np.where(y_test_bin == c)
        null_macro_mse += mean_squared_error(y_test_bin[idxs], macro_mse_trivial[idxs]) / len(train_classes)
        null_macro_vse += var_squared_error(y_test_bin[idxs], macro_mse_trivial[idxs]) / len(train_classes)
        null_macro_mae += mean_absolute_error(y_test_bin[idxs], macro_mae_trivial[idxs]) / len(train_classes)
        null_macro_vae += var_absolute_error(y_test_bin[idxs], macro_mae_trivial[idxs]) / len(train_classes)

    # Calculate AUCs
    if len(train_classes) > 2:
        auc = roc_auc_score(y_test_bin_binary, probs_bin, average='weighted')
    else:
        auc = roc_auc_score(y_test_bin_binary, probs_bin[:, 0], average='weighted')

    # Add scores
    scores = {'auc': auc,
              'mse': mse, 'vse': vse,
              'null_mse': null_mse, 'null_vse': null_vse,
              'mae': mae, 'vae': vae,
              'null_mae': null_mae, 'null_vae': null_vae,
              'macro_mse': macro_mse, 'macro_vse': macro_vse,
              'null_macro_mse': null_macro_mse, 'null_macro_vse': null_macro_vse,
              'macro_mae': macro_mae, 'macro_vae': macro_vae,
              'null_macro_mae': null_macro_mae, 'null_macro_vae': null_macro_vae}
    return scores


def generate_plots(results, filename, model_type, label_name):
    # Compute percent gains
    results['mse_percent_gain'] = (results['null_mse']-results['mse'])/results['null_mse']*100
    results['mae_percent_gain'] = (results['null_mae']-results['mae'])/results['null_mae']*100
    results['macro_mse_percent_gain'] = (results['null_macro_mse']-results['macro_mse'])/results['null_macro_mse']*100
    results['macro_mae_percent_gain'] = (results['null_macro_mae']-results['macro_mae'])/results['null_macro_mae']*100

    # Stack metrics for second plot
    results_plot = results.melt(id_vars='subject_id',
                                value_vars=["mse_percent_gain", "mae_percent_gain",
                                            "macro_mse_percent_gain", "macro_mae_percent_gain"])
    results_plot = results_plot.replace('mse_percent_gain', 'MSE')
    results_plot = results_plot.replace('mae_percent_gain', 'MAE')
    results_plot = results_plot.replace('macro_mse_percent_gain', 'Macro_MSE')
    results_plot = results_plot.replace('macro_mae_percent_gain', 'Macro_MAE')

    # Compute means and CIs
    auc_mean, auc_stderr = compute_mean_ci(results.auc)
    mse_mean, mse_stderr = compute_mean_ci(results.mse)
    mae_mean, mae_stderr = compute_mean_ci(results.mae)
    macro_mse_mean, macro_mse_stderr = compute_mean_ci(results.macro_mse)
    macro_mae_mean, macro_mae_stderr = compute_mean_ci(results.macro_mae)
    mse_percent_gain_mean, mse_percent_gain_stderr = compute_mean_ci(results.mse_percent_gain)
    mae_percent_gain_mean, mae_percent_gain_stderr = compute_mean_ci(results.mae_percent_gain)
    macro_mse_percent_gain_mean, macro_mse_percent_gain_stderr = compute_mean_ci(results.macro_mse_percent_gain)
    macro_mae_percent_gain_mean, macro_mae_percent_gain_stderr = compute_mean_ci(results.macro_mae_percent_gain)

    # Create titles
    title1 = 'Model: %s, Label: %s\n' % (model_type, label_name)
    title1 += 'AUC = %0.2f±%0.2f' % (auc_mean, auc_stderr)

    title2 = 'MSE = %0.2f±%0.2f, ' \
             'MAE = %0.2f±%0.2f, ' % \
             (mse_mean, mse_stderr,
              mae_mean, mae_stderr)
    title2 += 'Macro MSE = %0.2f±%0.2f, ' \
              'Macro MAE = %0.2f±%0.2f\n' % \
              (macro_mse_mean, macro_mse_stderr,
               macro_mae_mean, macro_mae_stderr)
    title2 += 'MSE %%Gain = %0.2f±%0.2f, ' \
              'MAE %%Gain = %0.2f±%0.2f, ' % \
              (mse_percent_gain_mean, mse_percent_gain_stderr,
               mae_percent_gain_mean, mae_percent_gain_stderr)
    title2 += 'Macro MSE %%Gain = %0.2f±%0.2f, ' \
              'Macro MAE %%Gain = %0.2f±%0.2f' % \
              (macro_mse_percent_gain_mean, macro_mse_percent_gain_stderr,
               macro_mae_percent_gain_mean, macro_mae_percent_gain_stderr)

    # Create x-ticks
    data_quantity = results[['subject_id', 'n_total']].drop_duplicates()
    x_ticks = ['%s (%d)' % (subj, quant) for subj, quant in
               zip(data_quantity.subject_id.values, data_quantity.n_total.values)]

    # Plot boxplot of AUCs
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(211)
    sns.boxplot(x='subject_id', y='auc', data=results)
    plt.axhline(0.5, 0, len(x_ticks), color='k', linestyle='--')
    plt.title(title1)
    ax.set_xticks([], [])
    plt.ylabel('AUC'), plt.ylim(0, 1)
    for x in np.arange(0, len(x_ticks), 1):
        plt.axvline(x + 0.5, -100, 100, color='k', linestyle='--')

    # Plot boxplot of MSE/MAE/etc
    ax = fig.add_subplot(212)
    sns.boxplot(x='subject_id', y='value', data=results_plot, hue='variable')
    plt.axhline(0, 0, len(x_ticks), color='k', linestyle='--')
    plt.title(title2)
    ax.set_xticklabels(x_ticks), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID (#samples)'), plt.ylabel('Percent Gain (Null - Model)/Null')
    for x in np.arange(0, len(x_ticks), 1):
        plt.axvline(x + 0.5, -100, 100, color='k', linestyle='--')

    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def var_squared_error(y1, y2):
    return np.var((y1-y2)**2)


def var_absolute_error(y1, y2):
    return np.var(np.abs(y1-y2))


def compute_mean_ci(x):
    mean_x = np.mean(x)
    stderr_x = scipy.stats.sem(x)
    return mean_x, stderr_x


def print_debug(text):
    if DEBUG:
        print(text)

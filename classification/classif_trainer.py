from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import label_binarize
import mord
from classification.ordinal_models.ordinal_rf import OrdinalRandomForestClassifier
import xgboost as xgb
import scipy.stats
from settings import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_user_model(data, label_name, model_type):
    print('Model:', model_type, ', Label:', label_name)
    filename = os.path.join('../figs/classification', '%s_%s.png' % (model_type, label_name))
    if os.path.exists(filename):
        return

    ground_truths, preds = np.array([]), np.array([])
    data_quantity = pd.DataFrame(columns=['subject_id', 'samples'])
    scores = pd.DataFrame(columns=['subject_id', 'AUC',
                                   'MSE', 'MAE', 'MSE_gain', 'MAE_gain',
                                   'Macro_MSE', 'Macro_MAE', 'Macro_MSE_gain', 'Macro_MAE_gain'])
    sorted_subjects = sorted(data.subject_id.unique())
    for subject in sorted_subjects[:2]:
        print_debug('--------------')
        print('Subject: %s' % subject)

        # Get data belonging to a specific subject
        subj_data = data[data.subject_id == subject].copy()
        subj_data.sort_values(by='timestamp', inplace=True)

        # Remove cases where on_off is not labeled
        subj_data = subj_data[subj_data.on_off > -1]

        # Make a table that just has unique measurement_ids and labels for the user
        id_table = subj_data[['ID', label_name]].drop_duplicates()
        data_quantity = data_quantity.append({'subject_id': subject, 'samples': len(id_table)}, ignore_index=True)

        # Remove any classes with not enough samples
        label_counts = id_table[label_name].value_counts()
        for i in range(len(label_counts)):
            if i in label_counts and label_counts[i] <= MIN_OBSERVATIONS_PER_CLASS:
                subj_data = subj_data[subj_data[label_name] != i]
                id_table = id_table[id_table[label_name] != i]
                print_debug('Removing class %d from this user' % i)

        # Skip if not enough data left over
        if len(id_table) <= MIN_OBSERVATIONS_PER_SUBJECT:
            print_debug('Not enough data points for that subject')
            continue

        rskf = RepeatedStratifiedKFold(n_splits=NUM_STRATIFIED_FOLDS, n_repeats=NUM_STRATIFIED_ROUNDS, random_state=RANDOM_SEED)
        for fold_idx, (train_idxs, test_idxs) in enumerate(list(rskf.split(id_table.ID, id_table[label_name]))):
            print_debug('Round: %d, Fold %d' % (int(fold_idx/NUM_STRATIFIED_FOLDS)+1,
                                                (fold_idx % NUM_STRATIFIED_FOLDS)+1))

            # Get measurement_ids for each fold
            id_train_set = id_table.ID.values[train_idxs]
            id_test_set = id_table.ID.values[test_idxs]

            # Separate train and test
            subj_data_train = subj_data[subj_data['ID'].isin(id_train_set)]
            subj_data_test = subj_data[subj_data['ID'].isin(id_test_set)]
            id_test = subj_data_test.ID.values

            # Separate into features and labels
            x_train = subj_data_train.iloc[:, :-7].values
            x_test = subj_data_test.iloc[:, :-7].values
            y_train = subj_data_train[label_name].values
            y_test = subj_data_test[label_name].values
            train_classes, test_classes = np.unique(y_train), np.unique(y_test)
            y_train = y_train.astype(np.int)
            y_test = y_test.astype(np.int)

            # Make sure that folds don't cut the data in a weird way
            if len(train_classes) <= 1:
                print_debug('Not enough classes in train')
                continue
            if len(test_classes) <= 1:
                print_debug('Not enough classes in test')
                continue
            if any([c not in train_classes for c in test_classes]):
                print_debug('There is a test class that is not in train')
                continue

            # Pick the correct model
            missing_class = any([k != train_classes[k] for k in range(len(train_classes))])
            if model_type == CLASSIF_RANDOM_FOREST:
                model = RandomForestClassifier(random_state=RANDOM_SEED)
                param_grid = dict(n_estimators=np.arange(10, 51, 10))
            elif model_type == CLASSIF_XGBOOST:
                model = xgb.XGBClassifier(objective="multi:softprob", random_state=RANDOM_SEED)
                model.set_params(**{'num_class': len(train_classes)})
                param_grid = dict(n_estimators=np.arange(80, 121, 20))
            elif model_type == CLASSIF_ORDINAL_RANDOM_FOREST:
                model = OrdinalRandomForestClassifier(random_state=RANDOM_SEED)
                param_grid = dict(n_estimators=np.arange(10, 51, 10))
            elif model_type == CLASSIF_ORDINAL_LOGISTIC:
                model = mord.LogisticSE()
                param_grid = dict(alpha=np.logspace(0, 3, 1))

                # Remap classes to fill in gap if one exists
                if missing_class:
                    print_debug('Forced to remap labels')
                    y_train = np.array(list(map(lambda x: np.where(train_classes == x), y_train))).flatten()
            elif model_type == CLASSIF_MLP:
                model = MLPClassifier(max_iter=1e3, random_state=RANDOM_SEED)
                num_features = x_train.shape[1]
                half_x, quart_x = int(num_features/2), int(num_features/4)
                param_grid = dict(hidden_layer_sizes=[(half_x), (half_x, quart_x)])
            else:
                raise Exception('Not a valid model type')

            # Identify ideal parameters using stratified k-fold cross-validation
            cross_validator = StratifiedKFold(n_splits=PARAM_SEARCH_FOLDS, random_state=RANDOM_SEED)
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=cross_validator)
            grid_search.fit(x_train, y_train)
            model.set_params(**grid_search.best_params_)
            print_debug('Done cross-validating')

            # Fit the model and predict classes
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            probs = model.predict_proba(x_test)
            lab = train_classes

            # If doing ordinal logistic regression, map classes back
            if model_type == CLASSIF_ORDINAL_LOGISTIC and missing_class:
                pred = np.array(list(map(lambda x: train_classes[x], pred))).flatten()
                new_probs = np.zeros(shape=(probs.shape[0], np.max(train_classes)+1))
                for c in train_classes:
                    new_probs[:, c] = probs[:, np.where(train_classes == c)].flatten()

            # Concatenate results
            ground_truths = np.concatenate([ground_truths, y_test])
            preds = np.concatenate([preds, pred])

            # Bin probabilities over each diary entry
            probs_bin = []
            y_test_bin = []
            pred_bin = []
            for ID in np.unique(id_test):
                probs_bin.append(np.mean(probs[id_test == ID, :], axis=0).reshape([1, -1]))
                y_test_bin.append(np.mean(y_test[id_test == ID]))
                pred_bin.append(np.mean(pred[id_test == ID]))
            probs_bin = np.vstack(probs_bin)
            y_test_bin = np.vstack(y_test_bin)
            pred_bin = np.vstack(pred_bin)

            # Binarize the results
            y_test_binary = label_binarize(y_test, lab)
            y_test_bin_binary = label_binarize(y_test_bin, lab)

            # Drop probabilities for classes not found in test data
            for i in list(range(np.shape(y_test_binary)[1]))[::-1]:
                if not any(y_test_bin_binary[:, i]):
                    probs = np.delete(probs, i, axis=1)
                    probs_bin = np.delete(probs_bin, i, axis=1)
                    y_test_binary = np.delete(y_test_binary, i, axis=1)
                    y_test_bin_binary = np.delete(y_test_bin_binary, i, axis=1)

            # Calculate MSE/MAE
            mse = mean_squared_error(y_test_bin, pred_bin)
            mae = mean_absolute_error(y_test_bin, pred_bin)

            # Compute null model MSE/MAE and the gain
            mse_trivial = np.ones(pred_bin.shape) * np.mean(y_train)
            mae_trivial = np.ones(pred_bin.shape) * np.median(y_train)
            null_model_mse = mean_squared_error(y_test_bin, mse_trivial)
            null_model_mae = mean_absolute_error(y_test_bin, mae_trivial)
            mse_gain = mse - null_model_mse
            mae_gain = mae - null_model_mae

            # Compute macro-MSE/MAE
            macro_mse, macro_mae = 0, 0
            for c in train_classes:
                idxs = np.where(y_test_bin == c)
                macro_mse += mean_squared_error(y_test_bin[idxs], pred_bin[idxs])/len(train_classes)
                macro_mae += mean_absolute_error(y_test_bin[idxs], pred_bin[idxs])/len(train_classes)

            # Compute null model macro-MSE/macro-MAE and the gain
            null_model_macro_mse, null_model_macro_mae = 0, 0
            macro_mse_trivial = np.ones(pred_bin.shape) * np.mean(train_classes)
            macro_mae_trivial = np.ones(pred_bin.shape) * np.median(train_classes)
            for c in train_classes:
                idxs = np.where(y_test_bin == c)
                null_model_macro_mse += mean_squared_error(y_test_bin[idxs], macro_mse_trivial[idxs]) / len(train_classes)
                null_model_macro_mae += mean_absolute_error(y_test_bin[idxs], macro_mae_trivial[idxs]) / len(train_classes)
            macro_mse_gain = macro_mse - null_model_macro_mse
            macro_mae_gain = macro_mae - null_model_macro_mae

            # Calculate AUCs
            if len(lab) > 2:
                auc = roc_auc_score(y_test_bin_binary, probs_bin, average='weighted')
            else:
                auc = roc_auc_score(y_test_bin_binary, probs_bin[:, 0], average='weighted')
            print_debug('AUC: %0.2f' % auc)

            # Add scores
            scores = scores.append({'subject_id': subject, 'AUC': auc,
                                    'MSE': mse, 'MAE': mae,
                                    'MSE_gain': mse_gain, 'MAE_gain': mae_gain,
                                    'Macro_MSE': macro_mse, 'Macro_MAE': macro_mae,
                                    'Macro_MSE_gain': macro_mse_gain, 'Macro_MAE_gain': macro_mae_gain},
                                   ignore_index=True)

    # Compute means and CIs
    auc_mean, auc_stderr = compute_mean_ci(scores.AUC)
    mse_mean, mse_stderr = compute_mean_ci(scores.MSE)
    mae_mean, mae_stderr = compute_mean_ci(scores.MAE)
    macro_mse_mean, macro_mse_stderr = compute_mean_ci(scores.Macro_MSE)
    macro_mae_mean, macro_mae_stderr = compute_mean_ci(scores.Macro_MAE)
    mse_gain_mean, mse_gain_stderr = compute_mean_ci(scores.MSE_gain)
    mae_gain_mean, mae_gain_stderr = compute_mean_ci(scores.MAE_gain)
    macro_mse_gain_mean, macro_mse_gain_stderr = compute_mean_ci(scores.Macro_MSE_gain)
    macro_mae_gain_mean, macro_mae_gain_stderr = compute_mean_ci(scores.Macro_MAE_gain)

    # Stack MSE/MAE for second plot 
    scores_plot = scores.copy()
    scores_plot = scores_plot.melt(id_vars='subject_id',
                                   value_vars=["MSE_gain", "MAE_gain", "Macro_MSE_gain", "Macro_MAE_gain"])
    scores_plot = scores_plot.replace("MSE_gain", "MSE")
    scores_plot = scores_plot.replace("MAE_gain", "MAE")
    scores_plot = scores_plot.replace("Macro_MSE_gain", "Macro_MSE")
    scores_plot = scores_plot.replace("Macro_MAE_gain", "Macro_MAE")

    # Create titles
    title1 = 'Model: %s, Label: %s\n' % (model_type, label_name)
    title1 += 'AUC = %0.2f±%0.2f' % (auc_mean, auc_stderr)

    title2 = 'Model: %s, Label: %s\n' % (model_type, label_name)
    title2 += 'MSE = %0.2f±%0.2f, ' \
              'MAE = %0.2f±%0.2f' % \
              (mse_mean, mse_stderr,
               mae_mean, mae_stderr)
    title2 += 'Macro MSE = %0.2f±%0.2f, ' \
              'Macro MAE = %0.2f±%0.2f\n' % \
              (macro_mse_mean, macro_mse_stderr,
               macro_mae_mean, macro_mae_stderr)
    title2 += 'MSE Gain = %0.2f±%0.2f, ' \
              'MAE Gain = %0.2f±%0.2f' % \
              (mse_gain_mean, mse_gain_stderr,
               mae_gain_mean, mae_gain_stderr)
    title2 += 'Macro MSE Gain = %0.2f±%0.2f, ' \
              'Macro MAE Gain = %0.2f±%0.2f\n' % \
              (macro_mse_gain_mean, macro_mse_gain_stderr,
               macro_mae_gain_mean, macro_mae_gain_stderr)

    # Create x-ticks
    x_ticks = ['%d (%d)' % (subj, quant) for subj, quant in zip(data_quantity.subject_id.values, data_quantity.samples.values)
               if subj in scores.subject_id.values]

    # Plot boxplot of AUCs
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    sns.boxplot(x='subject_id', y='AUC', data=scores)
    plt.axhline(0.5, 0, len(sorted_subjects), color='k', linestyle='--')
    plt.title(title1)
    ax.set_xticklabels(x_ticks), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID (#samples)')
    plt.ylabel('AUC'), plt.ylim(0, 1)
    for x in np.arange(0, len(sorted_subjects), 1):
        plt.axvline(x+0.5, -100, 100, color='k', linestyle='--')

    ax = fig.add_subplot(212)
    sns.boxplot(x='subject_id', y='value', data=scores_plot, hue='variable')
    plt.axhline(0, 0, len(sorted_subjects), color='k', linestyle='--')
    plt.title(title2)
    ax.set_xticklabels(x_ticks), plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.xlabel('Subject ID (#samples)'), plt.ylabel('Gain (Model - Null)')
    for x in np.arange(0, len(sorted_subjects), 1):
        plt.axvline(x+0.5, -100, 100, color='k', linestyle='--')

    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    print('**********************')


def print_debug(text):
    if DEBUG:
        print(text)


def compute_mean_ci(x):
    mean_x = np.mean(x)
    stderr_x = scipy.stats.sem(x)
    # ci = (mean_x-1.98*stderr_x, mean_x+1.98*stderr_x)
    return mean_x, stderr_x

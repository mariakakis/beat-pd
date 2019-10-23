from settings import *
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from model_training.ordinal_rf import OrdinalRandomForestClassifier
from model_training.helpers import calculate_scores, generate_plots, print_debug
import mord
import xgboost as xgb
import scipy.stats

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_user_classification(data, label_name, model_type, splits, run_id):
    print('Model:', model_type, ', Label:', label_name)
    image_filename = os.path.join(HOME_DIRECTORY, 'output', run_id, '%s_%s.png' % (model_type, label_name))
    csv_filename = os.path.join(HOME_DIRECTORY, 'output', run_id, '%s_%s.csv' % (model_type, label_name))
    if os.path.exists(image_filename):
        return

    results = pd.DataFrame(columns=['subject_id', 'n_total', 'n_train', 'n_test', 'auc',
                                    'mse', 'vse', 'null_mse', 'null_vse',
                                    'mae', 'vae', 'null_mae', 'null_vae',
                                    'macro_mse', 'macro_vse', 'null_macro_mse', 'null_macro_vse',
                                    'macro_mae', 'macro_vae', 'null_macro_mae', 'null_macro_vae'])
    sorted_subjects = sorted(data.subject_id.unique())
    if DEBUG:
        sorted_subjects = sorted_subjects[:5]

    for subject in sorted_subjects:
        print_debug('--------------')
        print('Subject: %s' % subject)

        # Get data belonging to a specific subject
        subj_data = data[data.subject_id == subject].copy()
        subj_data.sort_values(by='timestamp', inplace=True)

        # Remove cases without a label
        subj_data = subj_data[~np.isnan(subj_data[label_name])]
        subj_data = subj_data[subj_data[label_name] >= 0]

        # Make a table that just has unique measurement_ids and labels for the user
        id_table = subj_data[['ID', label_name]].drop_duplicates()

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

        # Create folds if needed
        rskf = RepeatedStratifiedKFold(n_splits=NUM_STRATIFIED_FOLDS,
                                       n_repeats=NUM_STRATIFIED_ROUNDS, random_state=RANDOM_SEED)
        folds = list(rskf.split(id_table.ID, id_table[label_name])) if splits is None else splits
        for fold_idx, (train_idxs, test_idxs) in enumerate(folds):
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
            x_train = subj_data_train.iloc[:, :-6].values
            x_test = subj_data_test.iloc[:, :-6].values
            y_train = subj_data_train[label_name].values.astype(np.int)
            y_test = subj_data_test[label_name].values.astype(np.int)
            train_classes, test_classes = np.unique(y_train), np.unique(y_test)

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

            # Construct the automatic feature selection method
            feature_selection = SelectPercentile(mutual_info_classif)
            param_grid = {'featsel__percentile': np.arange(25, 101, 25)}

            # Construct the base model
            missing_class = any([k != train_classes[k] for k in range(len(train_classes))])
            if model_type == CLASSIF_RANDOM_FOREST:
                base_model = RandomForestClassifier(random_state=RANDOM_SEED)
                param_grid = {'model__n_estimators': np.arange(10, 51, 10), **param_grid}
            elif model_type == CLASSIF_XGBOOST:
                base_model = xgb.XGBClassifier(objective="multi:softprob", random_state=RANDOM_SEED)
                base_model.set_params(**{'num_class': len(train_classes)})
                param_grid = {'model__n_estimators': np.arange(25, 76, 10), **param_grid}
            elif model_type == CLASSIF_ORDINAL_RANDOM_FOREST:
                base_model = OrdinalRandomForestClassifier(random_state=RANDOM_SEED)
                param_grid = {'model__n_estimators': np.arange(10, 51, 10), **param_grid}
            elif model_type == CLASSIF_ORDINAL_LOGISTIC:
                base_model = mord.LogisticSE()
                param_grid = {'model__alpha': np.logspace(-1, 1, 3), **param_grid}
            elif model_type == CLASSIF_MLP:
                base_model = MLPClassifier(max_iter=1000, random_state=RANDOM_SEED)
                num_features = x_train.shape[1]
                half_x, quart_x = int(num_features/2), int(num_features/4)
                param_grid = {'model__hidden_layer_sizes': [(half_x), (half_x, quart_x)], **param_grid}
            else:
                raise Exception('Not a valid model type')

            # Create a pipeline
            pipeline = Pipeline([
                ('featsel', feature_selection),
                ('model', base_model)
            ])

            # Remap classes to fill in gap if one exists
            if model_type in (CLASSIF_ORDINAL_RANDOM_FOREST, CLASSIF_ORDINAL_LOGISTIC) \
                    and missing_class:
                print_debug('Forced to remap labels')
                y_train = np.array(list(map(lambda x: np.where(train_classes == x), y_train))).flatten()

            # Identify ideal parameters using stratified k-fold cross-validation
            cross_validator = StratifiedKFold(n_splits=PARAM_SEARCH_FOLDS, random_state=RANDOM_SEED)
            grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cross_validator)
            grid_search.fit(x_train, y_train)
            model = pipeline.set_params(**grid_search.best_params_)
            print('Best params:', grid_search.best_params_)

            # Fit the model and predict classes
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            probs = model.predict_proba(x_test)

            # Calculate scores and other subject information
            scores = calculate_scores(y_train, y_test, train_classes, test_classes, id_test, preds, probs)
            result = {'subject_id': subject, 'n_total': len(train_idxs)+len(test_idxs),
                      'n_train': len(train_idxs), 'n_test': len(test_idxs),
                      **scores}
            results = results.append(result, ignore_index=True)

    # Save results
    results.to_csv(csv_filename, index=False, encoding='utf-8')

    # Plot results
    generate_plots(results, image_filename, model_type, label_name)
    print('**********************')
    return csv_filename, image_filename


def compute_mean_ci(x):
    mean_x = np.mean(x)
    stderr_x = scipy.stats.sem(x)
    # ci = (mean_x-1.98*stderr_x, mean_x+1.98*stderr_x)
    return mean_x, stderr_x

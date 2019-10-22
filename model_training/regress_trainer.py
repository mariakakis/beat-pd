from sklearn.exceptions import DataConversionWarning
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from settings import *
from model_training.helpers import calculate_scores, generate_plots

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


def train_user_regression(data, label_name, model_type):
    print('Model:', model_type, ', Label:', label_name)
    image_filename = os.path.join(HOME_DIRECTORY, 'output', 'regression', '%s_%s.png' % (model_type, label_name))
    csv_filename = os.path.join(HOME_DIRECTORY, 'output', 'regression', '%s_%s.csv' % (model_type, label_name))
    if os.path.exists(image_filename):
        return

    results = pd.DataFrame(columns=['subject_id', 'n_total', 'n_train', 'n_test', 'auc',
                                    'mse', 'vse', 'null_mse', 'null_vse',
                                    'mae', 'vae', 'null_mae', 'null_vae',
                                    'macro_mse', 'macro_vse', 'null_macro_mse', 'null_macro_vse',
                                    'macro_mae', 'macro_vae', 'null_macro_mae', 'null_macro_vae'])
    sorted_subjects = sorted(data.subject_id.unique())
    if DEBUG:
        sorted_subjects = sorted_subjects[:2]

    for subject in sorted_subjects:
        print_debug('--------------')
        print('Subject: %s' % subject)

        # Get data belonging to a specific subject
        subj_data = data[data.subject_id == subject].copy()
        subj_data.sort_values(by='timestamp', inplace=True)

        # Remove cases where on_off is not labeled
        subj_data = subj_data[subj_data.on_off > -1]

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
            feature_selection = SelectPercentile(mutual_info_regression)
            param_grid = {'featsel__percentile': np.arange(25, 101, 25)}

            # Construct the base model
            if model_type == REGRESS_XGBOOST:
                base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED)
                param_grid = {'model__n_estimators': np.arange(25, 76, 10), **param_grid}
            elif model_type == REGRESS_MLP:
                base_model = MLPRegressor(max_iter=1000, random_state=RANDOM_SEED)
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

            # Identify ideal parameters using stratified k-fold cross-validation
            cross_validator = StratifiedKFold(n_splits=PARAM_SEARCH_FOLDS, random_state=RANDOM_SEED)
            grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cross_validator)
            grid_search.fit(x_train, y_train)
            model = pipeline.set_params(**grid_search.best_params_)
            print('Best params:', grid_search.best_params_)

            # Fit the model and predict classes
            model.fit(x_train, y_train)
            preds = model.predict(x_test)

            # Compute probs from predicted values
            probs = np.zeros((len(preds), len(train_classes)))
            for i, pred in enumerate(preds):
                prob_vec = np.zeros((len(train_classes),))
                if pred <= np.min(train_classes):
                    prob_vec[0] = 1
                elif pred >= np.max(train_classes):
                    prob_vec[-1] = 1
                elif pred in train_classes:
                    idx = np.where(train_classes == pred)[0]
                    prob_vec[idx] = 1
                else:
                    lower_class_idx = np.max(np.where(pred > train_classes)[0])
                    upper_class_idx = np.min(np.where(pred < train_classes)[0])
                    lower_class = train_classes[lower_class_idx]
                    upper_class = train_classes[upper_class_idx]
                    prob_vec[lower_class_idx] = upper_class-pred
                    prob_vec[upper_class_idx] = pred-lower_class
                probs[i, :] = prob_vec

            # Calculate scores and other subject information
            scores = calculate_scores(y_train, y_test, train_classes, test_classes, id_test, preds, probs)
            result = {'subject_id': subject, 'n_total': len(train_idxs) + len(test_idxs),
                      'n_train': len(train_idxs), 'n_test': len(test_idxs),
                      **scores}
            results = results.append(result, ignore_index=True)

    # Save results
    results.to_csv(csv_filename, index=False, encoding='utf-8')

    # Plot results
    generate_plots(results, image_filename, model_type, label_name)
    print('**********************')


def print_debug(text):
    if DEBUG:
        print(text)

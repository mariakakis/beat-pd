from settings import *
from sklearn.exceptions import DataConversionWarning
from sklearn.pipeline import Pipeline, make_union
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, MissingIndicator
from sklearn.neighbors import KNeighborsRegressor
from model_training.helpers import preprocess_data, calculate_scores, generate_plots, print_debug
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_user_regression(data, id_table, label_name, model_type, run_id):
    print('Model:', model_type, ', Label:', label_name)
    image_filename = os.path.join(HOME_DIRECTORY, 'output', run_id, '%s_%s.png' % (model_type, label_name))
    csv_filename = os.path.join(HOME_DIRECTORY, 'output', run_id, '%s_%s.csv' % (model_type, label_name))
    if os.path.exists(image_filename):
        return

    results = pd.DataFrame(columns=['subject_id', 'split_id', 'n_total', 'n_train', 'n_test', 'auc',
                                    'mse', 'vse', 'null_mse', 'null_vse',
                                    'mae', 'vae', 'null_mae', 'null_vae',
                                    'macro_mse', 'macro_vse', 'null_macro_mse', 'null_macro_vse',
                                    'macro_mae', 'macro_vae', 'null_macro_mae', 'null_macro_vae'])
    sorted_subjects = sorted(id_table.subject_id.unique())
    if DEBUG:
        sorted_subjects = sorted_subjects[:5]

    for subject in sorted_subjects:
        print_debug('--------------')

        # Filter subject's data and generate folds, skipping if not enough data
        subj_id_table, folds = preprocess_data(id_table, subject, label_name)
        if subj_id_table is None:
            continue

        # Go through the folds
        for fold_idx, (id_table_train_idxs, id_table_test_idxs) in enumerate(folds):
            print('Subject: %s Fold: %d' % (subject, fold_idx))

            # Separate train and test IDs
            subj_id_table_train = subj_id_table.iloc[id_table_train_idxs, :]
            subj_id_table_test = subj_id_table.iloc[id_table_test_idxs, :]
            id_train = subj_id_table_train['ID'].values
            id_test = subj_id_table_test['ID'].values

            # Grab corresponding data
            subj_data_train = data[data['ID'].isin(id_train)]
            subj_data_test = data[data['ID'].isin(id_test)]

            # Add labels to the data
            subj_data_train = pd.merge(subj_data_train, subj_id_table_train[['ID', label_name]], on='ID', how='left')
            subj_data_test = pd.merge(subj_data_test, subj_id_table_test[['ID', label_name]], on='ID', how='left')

            # Separate into (train, validation, test) (features, labels)
            x_train = subj_data_train.drop(['ID', label_name], axis=1).values
            y_train = subj_data_train[label_name].values.astype(np.int)
            x_test = subj_data_test.drop(['ID', label_name], axis=1).values
            y_test = subj_data_test[label_name].values.astype(np.int)
            # x_train, x_valid, y_train, y_valid = \
            #     train_test_split(x_train, y_train, test_size=FRAC_VALIDATION_DATA, stratify=y_train,
            #                      random_state=RANDOM_SEED)
            train_classes = np.unique(y_train)
            test_classes = np.unique(y_test)
            # valid_classes = np.unique(y_valid)
            num_features = x_train.shape[1]

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

            # Prepare data imputer for missing data
            imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=int(num_features/10)),
                                       add_indicator=True, random_state=RANDOM_SEED)

            # Construct the automatic feature selection method
            feature_selection = SelectPercentile(mutual_info_regression)
            # param_grid = {'featsel__percentile': np.arange(25, 101, 25)}
            param_grid = {'featsel__percentile': 75}

            # Construct the base model
            if model_type == REGRESS_XGBOOST:
                base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED)
                # param_grid = {'model__n_estimators': np.arange(25, 76, 10), **param_grid}
                param_grid = {'model__n_estimators': 50, **param_grid}
            elif model_type == REGRESS_MLP:
                base_model = MLPRegressor(max_iter=1000, random_state=RANDOM_SEED)
                half_x, quart_x = int(num_features/2), int(num_features/4)
                # param_grid = {'model__hidden_layer_sizes': [(half_x), (half_x, quart_x)], **param_grid}
                param_grid = {'model__hidden_layer_sizes': (half_x), **param_grid}
            else:
                raise Exception('Not a valid model type')

            # Create a pipeline
            pipeline = Pipeline([
                ('imputer', imputer),
                ('featsel', feature_selection),
                ('model', base_model)
            ])

            # Identify ideal parameters using stratified k-fold cross-validation on validation data
            # cross_validator = StratifiedKFold(n_splits=PARAM_SEARCH_FOLDS, random_state=RANDOM_SEED)
            # grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cross_validator)
            # grid_search.fit(x_valid, y_valid)
            # model = pipeline.set_params(**grid_search.best_params_)
            # print('Best params:', grid_search.best_params_)
            model = pipeline.set_params(**param_grid)

            # Fit the model on train data
            model.fit(x_train, y_train)

            # Predict results on test data
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
            scores = calculate_scores(y_train, y_test, train_classes, test_classes, subj_data_test, preds, probs)
            result = {'subject_id': subject, 'split_id': fold_idx, 'n_total': len(id_table_train_idxs)+len(id_table_test_idxs),
                      'n_train': len(id_table_train_idxs), 'n_test': len(id_table_test_idxs),
                      **scores}
            results = results.append(result, ignore_index=True)

    # Save results
    results.to_csv(csv_filename, index=False, encoding='utf-8')

    # Plot results
    generate_plots(results, image_filename, model_type, label_name)
    print('**********************')
    return csv_filename, image_filename

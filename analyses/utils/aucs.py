OUTER_ITER = 1000
INNER_ITER = 100
ASSERT_RATIO = 0.95

def get_x_y(matrix, cohort, sample_col='sample', as_matrix=True):
    df_with_benefit = cohort.as_dataframe()[["patient_id", "is_benefit"]]
    df_with_benefit.columns = ["sample", "is_benefit"]

    matrix[sample_col] = matrix.index
    matrix = matrix.merge(df_with_benefit[['sample', 'is_benefit']], left_on=sample_col, right_on='sample', how='right')
    matrix = matrix.set_index(sample_col)
    y = matrix['is_benefit']
    del matrix['is_benefit']
    X = matrix
    X = X.fillna(0.0)
    if as_matrix:
        return X.as_matrix(), y.as_matrix()
    return X, y

def get_rank_auc(df_mut_counts, df_clinical, sort_col='mut_count'):
    from sklearn.metrics import roc_auc_score
    df_mut_benefit = df_mut_counts.merge(
        df_clinical[['sample', 'is_benefit']], on='sample', how='right')
    df_mut_benefit = df_mut_benefit.sort(sort_col, ascending=True)
    return roc_auc_score(df_mut_benefit.is_benefit, df_mut_benefit[sort_col])

def run_baseline(cohort):
    df_mut_counts = cohort.as_dataframe(on=missense_snv_count)[["patient_id", "missense_snv_count"]]
    df_clinical.columns = ["sample", "mut_count"]
    df_clinical = cohort.as_dataframe()[["patient_id", "is_benefit"]]
    df_clinical.columns = ["sample", "is_benefit"]
    X, y = get_x_y(df_mut_counts, df_clinical)
    return X, y

def classify(X_vals, y_val_or_clinical, sort=False,
             tetrapeptides=False, count_signature=False, matrix_func=None):
    import random
    np.random.seed(1234)
    random.seed(1234)

    C_list = [10**-i for i in np.arange(-5, 5, 0.5)]
    model_param_grid = {'C': C_list}
    scale_param_grid = {'scaler': [None, StandardScaler()]}

    def model_fit(X_train, y_train, model_param_set, scale_param_set):
        regression_model = LogisticRegression(penalty='l1', solver='liblinear', **model_param_set)
        estimators =  [('regression', regression_model)]
        scaler = scale_param_set['scaler']
        if scaler is not None:
            estimators = [('scale', scaler)] + estimators
        model = Pipeline(estimators)
        model.fit(X_train, y_train)
        return model, regression_model

    def filter_x_y(X_pre, y_pre, samples, matrix_func, is_get_xy=True):
        if not matrix_func:
            return X_pre[samples], y_pre[samples]
        X, y = matrix_func(X_pre, y_pre, samples)
        if is_get_xy:
            X, y = get_x_y(X, y)
        return X, y

    def freeze_params(model_param_set, scale_param_set, X_choice):
        return (frozenset(sorted(model_param_set.items())),
                frozenset(sorted(scale_param_set.items())),
                X_choice)

    def unfreeze_params(key):
        model_param_set, scale_param_set, X_choice = key
        model_param_set = dict(model_param_set)
        scale_param_set = dict(scale_param_set)
        return (model_param_set, scale_param_set, X_choice)

    def is_single_class(arr):
        if type(arr) == pd.DataFrame:
            arr_benefit_indices = list(arr[arr.is_benefit == 1].is_benefit)
            arr_no_benefit_indices = list(arr[arr.is_benefit == 0].is_benefit)
        else:
            arr_benefit_indices = (arr == 1).nonzero()[0]
            arr_no_benefit_indices = (arr == 0).nonzero()[0]
        return len(arr_benefit_indices) == 0 or len(arr_no_benefit_indices) == 0

    def get_folder(samples, n_iter, train_ratio):
        n = len(samples)
        train_size = int(n * train_ratio)
        test_size = n - train_size
        assert train_size + test_size == n
        for i in range(n_iter):
            samples_train = choice(samples, train_size, replace=True)
            samples_test = [sample for sample in samples if sample not in samples_train]
            yield samples_train, samples_test

    def single_model(X_train, y_train, X_test, y_test,
                     model_param_set, scale_param_set,
                     params_to_score):
        if not sort:
            model, regression_model = model_fit(X_train, y_train, model_param_set, scale_param_set)
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = X_test[:, 0]
        return roc_auc_score(y_test, prob)

    def single_inner(inner_samples):
        params_to_score = defaultdict(list)
        folder = get_folder(inner_samples, n_iter=INNER_ITER, train_ratio=0.75)
        for inner_train_samples, inner_test_samples in folder:
            for X_choice, X_val in enumerate(X_vals):
                assert len(set(inner_train_samples).intersection(set(inner_test_samples))) == 0
                X_train, y_train = filter_x_y(
                    X_val, y_val_or_clinical, inner_train_samples, matrix_func)
                X_test, y_test = filter_x_y(
                    X_val, y_val_or_clinical, inner_test_samples, matrix_func)
                usable_columns = X_train.std(axis=0).nonzero()[0]
                X_train = X_train[:, usable_columns]
                X_test = X_test[:, usable_columns]
                if (is_single_class(y_train) or is_single_class(y_test) or
                    X_train.shape[1] == 0 or X_test.shape[1] == 0):
                    continue
                for model_param_set in ParameterGrid(model_param_grid):
                    for scale_param_set in ParameterGrid(scale_param_grid):
                        auc_score = single_model(X_train, y_train, X_test, y_test,
                                                 model_param_set, scale_param_set,
                                                 params_to_score)
                        params_to_score[freeze_params(
                            model_param_set, scale_param_set, X_choice)].append(auc_score)

        if not params_to_score:
            return None

        def round(score):
            return float("%.2f" % score)

        params_to_single_score = dict()
        for key, value in params_to_score.items():
            params_to_single_score[key] = round(np.mean(value))

        best_param = max(params_to_single_score, key=params_to_single_score.get)
        best_score = params_to_single_score[best_param]

        # Handle ties
        best_params = [params for (params, score) in
                       params_to_single_score.items() if score == best_score]

        # Tie-breaking C values
        if len(best_params) > 1:
            best_c_vals = [dict(item[0])['C'] for item in best_params]
            winning_c = min(best_c_vals)
            winning_params = [item for item in best_params if dict(item[0])['C'] == winning_c]
            # If we still have multiple choices aside from C values, just choose the first
            best_param = winning_params[0]

        return unfreeze_params(best_param)

    samples_to_indices = None
    if type(y_val_or_clinical) == pd.DataFrame:
        if y_val_or_clinical.index.name == 'sample':
            y_val_or_clinical['sample'] = y_val_or_clinical.index
        samples = list(y_val_or_clinical['sample'])
        indices_of_samples = range(len(y_val_or_clinical))
        samples_to_indices = dict(zip(samples, indices_of_samples))
    else:
        samples = range(len(y_val_or_clinical))
    outer_folder = get_folder(samples,
                              n_iter=OUTER_ITER, train_ratio=0.75)
    aucs = []
    scale_params = []
    model_params = []
    record_samples = []
    hash_samples = []
    shapes = []
    len_tetra_sigs = []
    for train_samples, test_samples in outer_folder:
        if samples_to_indices:
            train_samples_record = tuple([samples_to_indices[s] for s in train_samples])
            test_samples_record = tuple([samples_to_indices[s] for s in test_samples])
        else:
            train_samples_record = tuple(list(train_samples))
            test_samples_record = tuple(list(test_samples))
        hash_value = hash(tuple([train_samples_record, test_samples_record]))
        record_samples.append((train_samples_record, test_samples_record))
        hash_samples.append(hash_value)
        assert len(set(train_samples).intersection(set(test_samples))) == 0
        if not sort and not tetrapeptides:
            result = single_inner(train_samples)
            if not result:
                continue
            model_param_set, scale_param_set, X_choice = result
            scale_params.append(scale_param_set)
            model_params.append(model_param_set)
        else:
            assert len(X_vals) == 1
            X_choice = 0

        X = X_vals[X_choice]
        y = y_val_or_clinical
        X_train, y_train = filter_x_y(X, y, train_samples, matrix_func=matrix_func, is_get_xy=(not tetrapeptides))
        X_test, y_test = filter_x_y(X, y, test_samples, matrix_func=matrix_func, is_get_xy=(not tetrapeptides))
        shape = {'X_train': X_train.shape, 'y_train': y_train.shape, 'X_test': X_test.shape, 'y_test': y_test.shape}
        shapes.append(shape)
        if is_single_class(y_train) or is_single_class(y_test):
            continue

        if tetrapeptides:
            # Get the tetrapeptide signature from the training data (only)
            tetrapeptide_signature = get_signature_kmers(X_train, y_train)
            len_tetra_sigs.append(len(tetrapeptide_signature))

            # Which samples in the test set match the tetrapeptide signature?
            X_test = in_signature(X_test, tetrapeptide_signature, count_signature=count_signature)
            prob, y_test = get_x_y(X_test, y_test)
        else:
            usable_columns = X_train.std(axis=0).nonzero()[0]
            X_train = X_train[:, usable_columns]
            X_test = X_test[:, usable_columns]
            if not sort:
                model, _ = model_fit(X_train, y_train, model_param_set, scale_param_set)
                prob = model.predict_proba(X_test)[:, 1]
            else:
                prob = X_test[:, 0]
        aucs.append(roc_auc_score(y_test, prob))
    return aucs

def calculate_auc(cohort, tetrapeptides, count_signature):
    # X, y =
    # ASSERT_RATIO
    X, y = 
    classify(X_vals, y_val_or_clinical, sort=False,
             tetrapeptides=False, count_signature=False, matrix_func=None):

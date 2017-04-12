# Copyright (c) 2016. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

from os import path
import numpy as np
import pandas as pd
from numpy.random import choice
import random
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from copy import copy
from multiprocessing import Process, Queue

from cohorts.rounding import float_str

from .data import init_cohort
from .tetrapeptides import get_signature_tetrapeptides

OUTER_ITER = 1000
INNER_ITER = 100
ASSERT_RATIO = 0.95

class Data(object):
    def __init__(self, cohort, features):
        self.cohort = cohort
        self.features = features

    @property
    def patient_ids(self):
        return [patient.id for patient in self.cohort]

    def sample(self, patient_ids):
        filtered_features = self.features[self.features.patient_id.isin(patient_ids)]

        # Because we are sampling with replacement, we may have repeat patients
        filtered_cohort = copy(self.cohort)
        filtered_patients = []
        for patient_id in patient_ids:
            if patient_id in [patient.id for patient in self.cohort]:
                patient = copy(self.cohort.patient_from_id(patient_id))
                filtered_patients.append(patient)
        filtered_cohort.elements = filtered_patients
        return Data(cohort=filtered_cohort, features=filtered_features)

class Evaluator(object):
    def __init__(self, data):
        self.data = data
        self.process_limit = data.cohort.process_limit
        np.random.seed(1234)
        random.seed(1234)
        C_list = [10**-i for i in np.arange(-5, 5, 0.5)]
        model_param_grid = {'C': C_list}
        scale_param_grid = {'scaler': [None, StandardScaler()]}

    def model_fit(self, data_train, model_param_set, scale_param_set):
        regression_model = LogisticRegression(penalty='l1', solver='liblinear', **model_param_set)
        estimators =  [('regression', regression_model)]
        scaler = scale_param_set['scaler']
        if scaler is not None:
            estimators = [('scale', scaler)] + estimators
        model = Pipeline(estimators)
        model.fit(data_train.X, data_train.y)
        return model, regression_model

    def freeze_params(self, model_param_set, scale_param_set):
        return (frozenset(sorted(model_param_set.items())),
                frozenset(sorted(scale_param_set.items())))

    def unfreeze_params(self, key):
        model_param_set, scale_param_set = key
        model_param_set = dict(model_param_set)
        scale_param_set = dict(scale_param_set)
        return (model_param_set, scale_param_set)

    def is_single_class(self, arr):
        if type(arr) == pd.DataFrame:
            arr_benefit_indices = list(arr[arr.is_benefit == 1].is_benefit)
            arr_no_benefit_indices = list(arr[arr.is_benefit == 0].is_benefit)
        else:
            arr_benefit_indices = (arr == 1).nonzero()[0]
            arr_no_benefit_indices = (arr == 0).nonzero()[0]
        return len(arr_benefit_indices) == 0 or len(arr_no_benefit_indices) == 0

    def get_folder(self, cohort, n_iter, train_ratio):
        n = len(cohort)
        train_size = int(n * train_ratio)
        test_size = n - train_size
        assert train_size + test_size == n
        for i in range(n_iter):
            patient_ids = [patient.id for patient in cohort]
            patient_ids_train = choice(patient_ids, train_size, replace=True)
            patient_ids_test = [patient_id for patient_id in patient_ids if patient_id not in patient_ids_train]
            yield patient_ids_train, patient_ids_test

    def single_model(self, data_train, data_test,
                     model_param_set, scale_param_set,
                     params_to_score):
        model, regression_model = self.model_fit(data_train, model_param_set, scale_param_set)
        prob = model.predict_proba(data_test.X)[:, 1]
        return roc_auc_score(data_test.y, prob)

    def keep_usable_columns(self, data_train, data_test):
        usable_columns = data_train.X.std(axis=0).nonzero()[0]
        data_train.X = data_train.X[:, usable_columns]
        data_test.X = data_test.X[:, usable_columns]

    def run_single_outer(self, q, data, train_patient_ids, test_patient_ids):
        pass

    def single_inner(inner_samples):
        params_to_score = defaultdict(list)
        folder = get_folder(inner_samples, n_iter=INNER_ITER, train_ratio=0.75)
        for inner_train_samples, inner_test_samples in folder:
            # Training and test must be separate!
            assert len(set(inner_train_samples).intersection(set(inner_test_samples))) == 0
            data_train = data.filter(inner_train_samples)
            data_test = data.filter(inner_test_samples)
            self.keep_usable_columns(data_train, data_test)
            if data_train.is_single_class() or data_test.is_single_class():
                continue

            for model_param_set in ParameterGrid(model_param_grid):
                for scale_param_set in ParameterGrid(scale_param_grid):
                    auc_score = single_model(data_train, data_test,
                                             model_param_set, scale_param_set,
                                             params_to_score)
                    params_to_score[freeze_params(
                        model_param_set, scale_param_set)].append(auc_score)

        if not params_to_score:
            return None

        params_to_single_score = dict()
        for key, value in params_to_score.items():
            params_to_single_score[key] = float_str(np.mean(value))

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

    def calculate_aucs(self):
        outer_folder = self.get_folder(self.data.cohort,
                                       n_iter=OUTER_ITER, train_ratio=0.75)
        queue = Queue()
        aucs = []
        processes = []
        num_pops = 0
        for train_patient_ids, test_patient_ids in outer_folder:
            print("Len processes: %d" % len(processes))
            while len(processes) >= self.process_limit:
                print("Pop %d!" % num_pops)
                num_pops += 1
                args, process = processes.pop(0)
                from multiprocessing import TimeoutError
                try:
                    process.join(timeout=10)
                except TimeoutError:
                    print("Timeout error!")
                    process = Process(target=self.run_single_outer, args=args)
                    #process.start()
                    processes.append((args, process))
            print("Len processes post pop: %d" % len(processes))

            args = (queue, self.data, train_patient_ids, test_patient_ids)
            process = Process(target=self.run_single_outer, args=args)

            process.start()
            processes.append((args, process))

        for args, process in processes:
            process.start()
            process.join()

        print("Num total proc: %d" % len(processes))

        while not queue.empty():
            aucs.append(queue.get())
        return aucs

class TetrapeptideEvaluator(Evaluator):
    def __init__(self, data, count):
        Evaluator.__init__(self, data)
        self.count = count

    def run_single_outer(self, queue, data, train_patient_ids, test_patient_ids):
        # Training and test must be separate!
        assert len(set(train_patient_ids).intersection(set(test_patient_ids))) == 0

        data_train = data.sample(train_patient_ids)
        data_test = data.sample(test_patient_ids)
        y_pred, y_test = self.get_x_y_tetrapeptides(data_train, data_test)
        queue.put(roc_auc_score(y_test, y_pred))

    def get_x_y_tetrapeptides(self, train_data, test_data):
        """
        Steps
        - From the train data, grab the signature tetrapeptides (e.g. those with certain counts)
        - Then, looking at just the test data, see which patients have tetrapeptides in the signature set
        """
        signature_tetrapeptides = get_signature_tetrapeptides(train_data.cohort, train_data.features)

        # Indexed by patient, value is a set of all the kmers per patient
        df_test_kmers = test_data.features.groupby("patient_id").agg({"kmer": lambda x: set(x)})
        df_test_kmers.rename(columns={"kmer": "tetrapeptides"}, inplace=True)

        if not self.count:
            # Does the patient have any signature tetrapeptides?
            def intersect(tetrapeptides):
                return len(tetrapeptides.intersection(signature_tetrapeptides)) > 0
        else:
            # How many signature tetrapeptides does the patient have?
            def intersect(tetrapeptides):
                return len(tetrapeptides.intersection(signature_tetrapeptides))
        df_test_kmers["signature"] = df_test_kmers.tetrapeptides.apply(intersect)

        # signature and patient_id
        X = df_test_kmers[["signature"]]
        y = X.join(test_data.cohort.as_dataframe().set_index("patient_id"))[["benefit"]]
        return X.as_matrix(), y.as_matrix()

def bootstrap_mean_formatter(series, q=0.95):
    if q < 0 or q > 1:
        raise ValueError("Invalid q %0.2f, needs to be within [0, 1]" % q)
    q = q * 100
    value = series.mean()
    low = np.percentile(series, (100 - q) / 2.0)
    high = np.percentile(series, (q + ((100 - q) / 2.0)))
    return ("%s, %d%% CI (%s, %s)" % (float_str(value), int(q), float_str(low), float_str(high)))

def run_tetrapeptides(nejm=True, count=False, formatter=bootstrap_mean_formatter):
    if nejm:
        cohort = init_cohort(four_callers=False)
        assert len(cohort) == 64
    else:
        cohort = init_cohort(four_callers=True,
                             biopsy_time="pre",
                             non_discordant=True)
        assert len(cohort) == 33
    cohort.print_provenance = False
    df_tetrapeptides = cohort.load_tetrapeptides()
    data = Data(cohort=cohort, features=df_tetrapeptides)
    evaluator = TetrapeptideEvaluator(data=data, count=count)
    aucs = evaluator.calculate_aucs()
    print("AUC of %s, %d AUCs" % (formatter(series=pd.Series(aucs)),
                                  len(aucs)))

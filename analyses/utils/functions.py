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

from cohorts.functions import *
from .variant_filters import effect_expressed_filter, neoantigen_expressed_filter

@use_defaults
def expressed_nonsynonymous_snv_count(row, cohort, filter_fn, normalized_per_mb, **kwargs):
    patient_id = row["patient_id"]
    patient = cohort.patient_from_id(patient_id)
    if patient.tumor_sample is None:
        return np.nan

    def expressed_filter_fn(filterable_effect):
        assert filter_fn is not None, "filter_fn should never be None, but it is."
        return filter_fn(filterable_effect) and effect_expressed_filter(filterable_effect)
    return nonsynonymous_snv_count(row=row,
                                   cohort=cohort,
                                   filter_fn=expressed_filter_fn,
                                   normalized_per_mb=normalized_per_mb)

@use_defaults
def expressed_neoantigen_count(row, cohort, filter_fn, normalized_per_mb, **kwargs):
    patient_id = row["patient_id"]
    patient = cohort.patient_from_id(patient_id)
    if patient.tumor_sample is None:
        return np.nan

    def expressed_filter_fn(filterable_neoantigen):
        assert filter_fn is not None, "filter_fn should never be None, but it is."
        return filter_fn(filterable_neoantigen) and neoantigen_expressed_filter(filterable_neoantigen)
    return neoantigen_count(row=row,
                            cohort=cohort,
                            filter_fn=expressed_filter_fn,
                            normalized_per_mb=normalized_per_mb)

@count_function
def homologous_epitope_count(row, cohort, filter_fn, normalized_per_mb, **kwargs):
    patient_id = row["patient_id"]
    patient = cohort.patient_from_id(patient_id)
    try:
        df_epitopes = cohort.load_single_patient_epitope_homology(patient, **kwargs)
    except AssertionError as e:
        import pandas as pd
        # TODO REMOVE
        print(e)
        df_epitopes = pd.DataFrame({"patient_id": [patient_id]})

    return {patient_id: df_epitopes}

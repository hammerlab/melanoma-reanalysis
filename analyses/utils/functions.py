from cohorts.functions import *
from .variant_filters import effect_expressed_filter, neoantigen_expressed_filter

@use_defaults
def expressed_missense_snv_count(row, cohort, filter_fn, normalized_per_mb, **kwargs):
    patient_id = row["patient_id"]
    patient = cohort.patient_from_id(patient_id)
    if patient.tumor_sample is None:
        return np.nan

    def expressed_filter_fn(filterable_effect):
        assert filter_fn is not None, "filter_fn should never be None, but it is."
        return filter_fn(filterable_effect) and effect_expressed_filter(filterable_effect)
    return missense_snv_count(row=row,
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

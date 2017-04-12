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

from varcode import Variant
from varcode.common import memoize

@memoize
def expressed_variant_set(patient, cohort, variant_collection):
    df_expression = cohort.load_single_patient_expressed(patient)
    if df_expression is None:
        raise ValueError("No expression data found for patient %s" % patient.id)

    expressed_variant_set = set()
    for _, row in df_expression.iterrows():
        expressed_variant = Variant(contig=row["Chr"],
                                    start=row["Position"],
                                    ref=row["Ref"],
                                    alt=row["Alt"],
                                    ensembl=variant_collection[0].ensembl)
        expressed_variant_set.add(expressed_variant)

    return expressed_variant_set

def variant_expressed_filter(filterable_variant, **kwargs):
    patient = filterable_variant.patient
    expressed_variants = expressed_variant_set(
        filterable_variant.patient,
        filterable_variant.patient.cohort,
        filterable_variant.variant_collection)
    return filterable_variant.variant in expressed_variants

def effect_expressed_filter(filterable_effect, **kwargs):
    return variant_expressed_filter(filterable_effect, **kwargs)

def neoantigen_expressed_filter(filterable_neoantigen, **kwargs):
    return variant_expressed_filter(filterable_neoantigen, **kwargs)

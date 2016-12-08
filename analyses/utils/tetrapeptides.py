import numpy as np
import pandas as pd
from os import path
from collections import defaultdict
from topeology.iedb_data import get_iedb_epitopes, DataFilter

from .data import REPO_DATA_DIR

def get_tetrapeptides_peptides(cursor):
    from checkpoint_utils import patient_data
    from checkpoint_utils.compare import normalize_sample
    from checkpoint_utils.clinical_utils import is_benefit
    mutations = patient_data.load_nejm_mutations()
    mutations['sample'] = mutations['SampleId'].apply(normalize_sample)
    return mutations

def get_tetrapeptides(cohort):
    neoantigens = cohort.load_neoantigens()

    # Combine neoantigens from all patients
    df_neoantigens = pd.concat(neoantigens).reset_index(drop=True)

    # Get kmers from neoantigens
    df_kmers = join_with_kmers(
        df_neoantigens,
        kmer_size=4,
        peptide_column="peptide").dropna()

    df_iedb = get_iedb_epitopes(
        epitope_lengths=cohort.epitope_lengths,
        data_filters=[DataFilter(on="tuberculosis")],
        iedb_path=path.join(REPO_DATA_DIR, "iedb_tcell_data_6_10_15.csv"))
    indexed_epitopes = defaultdict(list)
    for _, row in df_iedb.iterrows():
        seq = row["iedb_epitope"]
        # Get kmers from IEDB epitopes
        substrings = get_kmers(seq, kmer_size=4)
        for sub in substrings:
            indexed_epitopes[sub] += [row]

    df_kmers["in_iedb"] = df_kmers["kmer"].isin(indexed_epitopes.keys())
    df_kmers = df_kmers.merge(cohort.as_dataframe(), on="patient_id", how="right")
    # There can be duplicates due to repeated tetrapeptides in one 9mer, repeated in one sample, etc.
    df_kmers = df_kmers[["patient_id", "peptide", "in_iedb", "kmer"]].reset_index(drop=True)
    return df_kmers

def join_with_kmers(df, kmer_size, peptide_column, result_column="kmer"):
    """
    Tag dataframe with kmers of kmer_size
    """
    def translate(peptide):
        kmers = []
        if not isinstance(peptide, float):
            kmers = get_kmers(peptide, kmer_size)
        return kmers

    serieses = []
    for i, row in df.iterrows():
        series = pd.Series(i, translate(row[peptide_column]))
        serieses.append(series)
    df_kmer_link = pd.concat(serieses).reset_index()

    df_kmer_link.columns = [result_column, "index"]
    df_kmer_link.set_index(["index"], inplace=True)

    return df.join(df_kmer_link, how="left")

def get_kmers(seq, kmer_size):
    """
    Split a sequence in kmer_size substrings
    """
    return [seq[i: i + kmer_size].upper()
                for i in range(0, len(seq) - kmer_size + 1)]

def get_signature_kmers(cohort, df_tetrapeptides):
    # TODO: Figure out why we need this and remove if possible
    assert len(df_tetrapeptides) == len(df_tetrapeptides.index)

    # TODO REMOVE?
    df_tetrapeptides["patient_id"] = df_tetrapeptides.index
    df_tetrapeptides = df_tetrapeptides.merge(cohort.as_dataframe(), on="patient_id")
    stats = df_kmers.groupby(["kmer"]).agg({
        "patient_id": [np.count_nonzero, pd.Series.nunique],
        "peptide": [np.count_nonzero, pd.Series.nunique],
        "in_iedb": [np.any],
        "benefit": [np.sum]
    })
    del df_kmers["patient_id"]
    N = 2
    signature = stats[
        # All appearances are benefit appearances
        (stats["patient_id"]["count_nonzero"] == stats["benefit"]["sum"]) &
        # > N discovery appearances
        ((stats["patient_id"]["nunique"] > N) |
         # >= N discovery appearances and in IEDB
         ((stats["patient"]["nunique"] >= N) & (stats["in_iedb"]["any"])))
    ]
    signature_kmers = set(signature.index)
    return signature_kmers

def in_signature(X_pre, signature_kmers, tetra_count):
    X_pre["signature"] = X_pre["kmer"].isin(signature_kmers)
    if tetra_count:
        return X_pre.groupby(level=0)[["signature"]].count()
    else:
        has_signature = X_pre.groupby(level=0)[["signature"]].any()
        has_signature.signature = has_signature.signature.apply(lambda x: 1.0 if x else 0.0)
        return has_signature

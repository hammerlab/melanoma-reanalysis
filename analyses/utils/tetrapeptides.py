import numpy as np

def run_tetrapeptides(cohort, func_clinical, func_peptides, peptide_column):
    def get_tetrapeptides(df_clinical):
        from checkpoint_utils import patient_data
        from checkpoint_utils.compare import normalize_sample
        from checkpoint_utils.clinical_utils import is_benefit
        epitopes = func_peptides(cursor)
        df_kmers = iedb.join_with_kmers(
                            epitopes, 
                            kmer_size = 4,
                            peptide_column = peptide_column).dropna()

        tcell_positive = iedb.load_tcell_positive('../../data/tcell_compact_addendum_6_10_15.csv')
        from collections import defaultdict
        indexed_epitopes = defaultdict(list)
        for _, epitope in tcell_positive.iterrows():
            seq = epitope['Epitope Linear Sequence']
            substrings = iedb.get_kmers(seq, kmer_size = 4)
            for sub in substrings:
                indexed_epitopes[sub] += [epitope]

        df_kmers['in_iedb'] = df_kmers['kmer'].isin(indexed_epitopes)
        df_kmers['is_benefit'] = df_kmers['sample'].apply(is_benefit)
        df_kmers = df_kmers.merge(df_clinical, on='sample', how='right')
        # There can be duplicates due to repeated tetrapeptides in one 9mer, repeated in one sample, etc.
        df_kmers = df_kmers[['sample', peptide_column, 'in_iedb', 'kmer']]
        return df_kmers

    df_clinical = func_clinical(cohort)
    X = get_tetrapeptides(df_clinical)
    y = df_clinical
    return X, y

def get_signature_kmers(df_kmers, df_clinical):
    assert len(df_kmers) == len(df_kmers.index)
    df_kmers['sample'] = df_kmers.index
    df_kmers = df_kmers.merge(df_clinical, on='sample')
    # TODO: Remove this hardcoding
    peptide_column = 'Mut peptide' if 'Mut peptide' in df_kmers.columns else 'peptide_mut'
    stats = df_kmers.groupby(['kmer']).agg({
        'sample': [np.count_nonzero, pd.Series.nunique],
        peptide_column: [np.count_nonzero, pd.Series.nunique],
        'in_iedb': [np.any],
        'is_benefit': [np.sum]
    })
    del df_kmers['sample']
    N = 2
    signature = stats[
        # All appearances are benefit appearances
        (stats['sample']['count_nonzero'] == stats['is_benefit']['sum']) &
        # > N discovery appearances
        ((stats['sample']['nunique'] > N) |
         # >= N discovery appearances and in IEDB
         ((stats['sample']['nunique'] >= N) & (stats['in_iedb']['any'])))
    ]
    signature_kmers = set(signature.index)
    return signature_kmers

def in_signature(X_pre, signature_kmers, tetra_count):
    X_pre['signature'] = X_pre['kmer'].isin(signature_kmers)
    if tetra_count:
        return X_pre.groupby(level=0)[['signature']].count()
    else:
        has_signature = X_pre.groupby(level=0)[['signature']].any()
        has_signature.signature = has_signature.signature.apply(lambda x: 1.0 if x else 0.0)
        return has_signature

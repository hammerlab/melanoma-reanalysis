import numpy as np

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

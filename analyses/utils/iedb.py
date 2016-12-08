from __future__ import print_function, absolute_import

from mhctools.alleles import normalize_allele_name, compact_allele_name

def get_iedb_binders():
    df_binding = pd.read_csv("../../data/iedb_binding_pred.csv")
    df_binders = df_binding[df_binding.value <= 500]
    df_binders = df_binders[['peptide', 'length', 'allele']].drop_duplicates()
    df_binders.columns = ['peptide_iedb', 'length', 'hla']
    from mhctools.alleles import compact_allele_name
    df_binders.hla = df_binders.hla.apply(compact_allele_name)
    df_binders = df_binders.reset_index(drop=True)
    return df_binders

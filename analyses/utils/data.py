import pandas as pd
from os import path, getcwd, environ
import numpy as np
from collections import defaultdict
from pyensembl import EnsemblRelease
from varcode import Variant, VariantCollection
from varlens.read_evidence import PileupCollection
from cohorts import Sample, Patient, Cohort, DataFrameLoader
from topeology import compare, DataFilter, iedb_data
from topeology.iedb_data import get_iedb_epitopes
from mhctools.alleles import compact_allele_name

def get_dir(env_var):
    dir = environ.get(env_var, None)
    if dir is None:
        raise ValueError("Must set environment variable %s" % env_var)
    return dir

# This repo's "data" directory
REPO_DATA_DIR = path.join(path.dirname(path.dirname(getcwd())), "data")

# The directory where we want to store cached mutations, neoantigens, etc,
CACHE_DATA_DIR = path.join(REPO_DATA_DIR, "cache")

BAM_DATA_DIR = get_dir("BAM_DATA_DIR")

def init_cohort(**kwargs):
    return MelanomaData(**kwargs).load_cohort()

class MelanomaData(object):
    """
    Represents the ipi/melanoma cohort.

    Parameters
    ----------
    join_with : list
        Which dataframes to join on by default.
    join_how : list
        How to join those dataframes by default.
    """
    def __init__(self,
                 four_callers=True,
                 biopsy_time=None,
                 non_discordant=False,
                 epitope_lengths=[8, 9, 10, 11],
                 repo_data_dir=REPO_DATA_DIR,
                 cache_data_dir=CACHE_DATA_DIR,
                 rna_bam_path=BAM_DATA_DIR):
        self.four_callers = four_callers
        self.biopsy_time = biopsy_time
        self.non_discordant = non_discordant
        self.epitope_lengths = epitope_lengths
        self.repo_data_dir = repo_data_dir
        self.rna_bam_path = rna_bam_path

        if not four_callers:
            self.cache_data_dir = cache_data_dir + "_nejm"
        else:
            self.cache_data_dir = cache_data_dir

    def load_cohort(self):
        benefit_ids = self.load_benefit_ids()
        discordant_ids = self.load_discordant_ids()
        hla_map = self.load_hla_map()
        rna_bam_map = self.load_rna_bam_map()
        patients_df = self.load_patients_df()
        patients = []

        if self.four_callers:
            df_mutations = self.load_four_caller_mutations()
            patient_id_to_variants = self.df_to_variant_collections(
                df=df_mutations,
                contig_col='CHROM',
                pos_col='POS',
                ref_col='REF',
                alt_col='ALT')
        else:
            df_mutations = self.load_nejm_mutations()
            patient_id_to_variants = self.df_to_variant_collections(
                df=df_mutations,
                contig_col='Chr',
                pos_col='Position',
                ref_col='Ref',
                alt_col='Alt')

        for i, row in patients_df.iterrows():
             # Skip post if filtering to pre
            if self.biopsy_time == "pre" and row["Biopsy pre or post ipi"] == "post":
                    continue
            # Skip pre if filtering to post
            elif self.biopsy_time == "post" and row["Biopsy pre or post ipi"] == "pre":
                    continue

            patient_id = self.normalize_id(row["Study ID"])

            # Skip discordant if filtering to non-discordant
            if self.non_discordant and patient_id in discordant_ids:
                continue

            if patient_id in rna_bam_map.keys():
                rna_bam_id = rna_bam_map[patient_id]
                tumor_sample = Sample(
                    is_tumor=True,
                    bam_path_rna=path.join(
                        self.rna_bam_path,
                        "Sample_%s-Sample_%s" % (rna_bam_id, rna_bam_id),
                        "Sample_%s-tumor-star-alignedAligned.sortedByCoord.out.bam" %  rna_bam_id),
                    cufflinks_path=path.join(
                        self.rna_bam_path,
                        "Sample_%s-Sample_%s" % (rna_bam_id, rna_bam_id),
                        "Sample_%s-cufflinks_output" % (rna_bam_id),
                        "genes.fpkm_tracking"
                    ))
            else:
                tumor_sample = None

            patient = Patient(id=patient_id,
                              os=row["OS"],
                              # Only using OS in this analysis.
                              pfs=row["OS"],
                              deceased=row["Alive"] != 1,
                              # Only using OS in this analysis.
                              progressed=row["Alive"] != 1,
                              benefit=patient_id in benefit_ids,
                              hla_alleles=hla_map[patient_id],
                              snv_variant_collection=patient_id_to_variants[patient_id],
                              tumor_sample=tumor_sample,
                              additional_data=row)
            patients.append(patient)

        # Add-in code specific to this analysis
        Cohort.load_single_patient_expressed = load_single_patient_expressed
        Cohort.load_single_patient_epitope_homology = load_single_patient_epitope_homology
        Cohort.load_single_allele_iedb_binders = load_single_allele_iedb_binders
        Cohort.load_iedb_binders = load_iedb_binders

        cohort = Cohort(patients,
                        cache_dir=self.cache_data_dir)
        cohort.epitope_lengths = self.epitope_lengths

        if not self.four_callers:
            def not_supported(self, **kwargs):
                raise NotImplementedError("This function is not implemented for the NEJM mutations")
            cohort.load_effects = not_supported
            cohort.load_neoantigens = not_supported

        # Make new, non-default caches for non-isovar expression and homology
        cohort.cache_names["expression"] = "cached-expression"
        cohort.cache_names["homology"] = "cached-epitope-homology"
        cohort.cache_names["iedb-binders"] = "cached-iedb-binders"
        return cohort

    def load_patient_ids(self):
        patients_df = self.load_patients_df()
        patients_df["patient_id"] = patients_df["Study ID"].map(self.normalize_id)
        return set(patients_df.patient_id.unique())

    def normalize_id(self, id):
        id = ''.join([char for char in id if (char.isdigit() or
                                              char == '_')])
        if '_' in id:
            parts = id.split('_')
            parts = [part for part in parts if part != '']
            id = parts[0]
        return id

    def load_rna_bam_map(self):
        df_rna_ids = pd.read_csv(
            path.join(self.repo_data_dir, "rna_bam_ids.csv"),
            dtype="object")
        df_rna_ids["patient_id"] = df_rna_ids["Study ID"].map(self.normalize_id)
        df_rna_ids = df_rna_ids[["patient_id", "RNA_FILE_ID"]]
        patient_id_to_rna_bam_id = dict(df_rna_ids.to_dict('split')['data'])
        return patient_id_to_rna_bam_id

    def load_discordant_ids(self):
        discordant_ids = [self.normalize_id(patient_id) for patient_id in [
            'CR7623',
            'CRNR0244',
            'CRNR2472',
            'CRNR4941',
            'LSDNR1120',
            'LSDNR1650',
            'LSDNR3086',
            'LSDNR9298',
            'PR03803']]
        assert len(discordant_ids) == 9, (
            "Incorrect number of IDs that are discordant: %d" % len(benefit_ids))
        # Make sure that all discordant IDs are actual patient IDs
        assert set(discordant_ids).union(self.load_patient_ids()) == self.load_patient_ids(), (
            "All discordant IDs must be actual patient IDs")
        return discordant_ids

    def load_benefit_ids(self):
        benefit_ids = [self.normalize_id(patient_id) for patient_id in [
            'SD2056', 'CR3665', 'CR0095', 'CR6336', 'CR1509', 'CR4880',
            'SD0346', 'SD1494', 'CR9699', 'CR9306', 'PR4046', 'LSD2057',
            'LSD0167', 'CR22640', 'LSD6819', 'PR4091', 'CR06670', 'CR6126',
            'PR4035', 'LSD4691', 'LSD3484', 'CR6161', 'PR12117', 'LSD4744',
            'PR4092', 'PR4077', 'CR04885']]
        assert len(benefit_ids) == 27, (
            "Incorrect number of IDs with benefit: %d" % len(benefit_ids))
        # Make sure that all benefit IDs are actual patient IDs
        assert set(benefit_ids).union(self.load_patient_ids()) == self.load_patient_ids(), (
            "All benefit IDs must be actual patient IDs")
        return benefit_ids

    def load_hla_map(self):
        df_hla_types = pd.read_csv(
            path.join(self.repo_data_dir, "updated_hla_types.csv"),
            dtype="object",
            quotechar='"')
        patient_id_to_hla_str = dict(df_hla_types.to_dict('split')['data'])
        patient_id_to_hla_list = {}
        for patient_id, hla_str in patient_id_to_hla_str.items():
            hla_list = hla_str.split(",")
            hla_list = [hla_type for hla_type in hla_list if hla_type != '']
            patient_id_to_hla_list[patient_id] = hla_list
        return patient_id_to_hla_list

    def load_patients_df(self):
        discovery_df = self.load_discovery_set()
        validation_df = self.load_validation_set()
        data_columns = ['Study ID',
                        'Age',
                        'Gender',
                        u'Primary Melanoma type',
                        u'M stage',
                        u'Priors^',
                        u'Ipi dosing (mg/kg x #)',
                        u'Response duration (weeks)',
                        u'LDH (normal <246)',
                        u'BRAF/NRASstatus',
                        u'OS',
                        u'Biopsy pre or post ipi',
                        'Alive']
        discovery_df.columns = data_columns
        validation_df.columns = data_columns
        patients = pd.concat([discovery_df, validation_df])
        return patients

    def load_discovery_set(self):
        # Load patient data
        DISCOVERY_SET_FILE = 'nejm_table_s1.csv'
        data = pd.read_csv(
            path.join(self.repo_data_dir, DISCOVERY_SET_FILE),
            quotechar='"')
        set_size = len(data)
        assert set_size == 25, \
            "Did not find correct number of discovery set patients, found {}".format(set_size)
        return data

    def load_validation_set(self):
        # Load patient data
        # Only uses PR12117; no need to fix typo here.
        VALIDATION_SET_FILE = 'nejm_table_s2.csv'
        data = pd.read_csv(
            path.join(self.repo_data_dir, VALIDATION_SET_FILE),
            quotechar='"')
        return data

    def df_to_variant_collections(self, df, contig_col='Chr',
                                  pos_col='Position', ref_col='Ref',
                                  alt_col='Alt', release_num=75):
        patient_id_to_variant_list = defaultdict(list)
        release = EnsemblRelease(release_num)
        for i, row in df.iterrows():
            contig = row[contig_col]
            start = int(row[pos_col])
            ref = str(row[ref_col])
            alt = str(row[alt_col])
            variant = Variant(contig=contig,
                              start=start,
                              ref=ref,
                              alt=alt,
                              ensembl=release)
            patient_id = row['patient_id']
            patient_id_to_variant_list[patient_id].append(variant)

        patient_id_to_variants = {}
        for patient_id in patient_id_to_variant_list.keys():
            variant_list = patient_id_to_variant_list[patient_id]
            patient_id_to_variants[patient_id] = VariantCollection(variant_list)
        return patient_id_to_variants

    def load_nejm_mutations(self):
        df = pd.read_csv(path.join(self.repo_data_dir, "nejm_table_s3.csv"))

        # Only use PR12117; the other is a typo.
        df["Sample"] = df['Sample'].map(lambda sample: sample if sample != 'PR11217' else 'PR12117')
        df["patient_id"] = df["Sample"].map(self.normalize_id)

        assert set(df.patient_id.unique()).union(self.load_patient_ids()) == self.load_patient_ids(), (
            "All IDs must be actual patient IDs")
        return df

    def load_four_caller_mutations(self, filter_first=True):
        def sample_number(sample):
            if '_' in sample:
                parts = sample.split('_')
                parts = [part for part in parts if part != '']
                if len(parts[-1]) == 1:
                    return int(parts[-1])
            return 1
        data = pd.read_csv(path.join(self.repo_data_dir, "four_caller_snvs.csv"))
        data["patient_id"] = data["Sample"].map(self.normalize_id)
        data["sample_number"] = data["Sample"].map(sample_number)

        # Remove extra patient not included in NEJM.
        data = data[data.patient_id != '8178']

        if filter_first:
            return data[data.sample_number == 1]

        assert set(data.patient_id.unique()).union(self.load_patient_ids()) == self.load_patient_ids(), (
            "All IDs must be actual patient IDs")
        return data

def load_single_patient_expressed(cohort, patient):
    expression_cached_file_name = "%s-%s-expression.csv" % (cohort.variant_type, cohort.merge_type)
    df_expression = cohort.load_from_cache(cohort.cache_names["expression"], patient.id, expression_cached_file_name)
    if df_expression is not None:
        return df_expression

    rows = []
    variants = cohort._load_single_patient_variants(patient, filter_fn=None)

    if patient.tumor_sample is None:
        raise ValueError("No expression data found for patient %s" % patient.id)

    pc = PileupCollection.from_bam(patient.tumor_sample.bam_path_rna, variants)
    for variant in variants:
        summary = dict(pc.match_summary(variant))
        row = {}
        row['Chr'] = variant.contig
        row['Position'] = variant.start
        row['Ref'] = variant.ref
        row['Alt'] = variant.alt
        row['RNARefReads'] = summary.get(variant.ref)
        row['RNAAltReads'] = summary.get(variant.alt)
        rna_depth = float(summary.get(variant.ref) + summary.get(variant.alt))
        row['RNA-VAF'] = summary.get(variant.alt) / rna_depth if rna_depth != 0 else np.nan
        row['IsExpressed'] = row['RNA-VAF'] > 0
        rows.append(row)
    df_all_expression = pd.DataFrame.from_records(rows)
    df_expression = df_all_expression[df_all_expression.IsExpressed]
    assert len(df_expression) != len(df_all_expression), "Some expressed variants must be filtered out"
    cohort.save_to_cache(df_expression, cohort.cache_names["expression"], patient.id, expression_cached_file_name)
    return df_expression

def load_single_patient_epitope_homology(cohort, patient, include_wildtype=False, include_organism=False):
    cached_file_name = "%s-%s-%s%shomology.csv" % (cohort.variant_type, cohort.merge_type, "wildtype-" if include_wildtype else "", "organism-" if include_organism else "")
    df_homology = cohort.load_from_cache(cohort.cache_names["homology"], patient.id, cached_file_name)
    if df_homology is not None:
        return df_homology

    neoantigens = cohort.load_neoantigens(patients=[patient])
    # Return all effects, not just top-priority effects, since all effect were used by topiary
    effects = cohort.load_effects(patients=[patient], all_effects=True)
    effect_collection = effects[patient.id]
    df_neoantigens = neoantigens[patient.id]

    # Only look at substitutions; because that's the goal, and also because the
    # wildtype logic doesn't apply to e.g. StopLoss
    df_neoantigens = df_neoantigens[df_neoantigens.effect_type == "Substitution"]

    def wildtype_peptide(row, genome=cohort.genome, effects=effect_collection):
        variant = Variant(
            contig=row["chr"],
            ref=row["ref"],
            alt=row["alt"],
            start=row["start"],
            ensembl=genome)
        effects = effects.filter(lambda effect: effect.variant == variant and
                                 effect.transcript.id == row["transcript_id"] and
                                 type(effect).__name__ == row["effect_type"])
        assert len(effects) == 1, "Expected a single effect to match the variant %s, but have %s for patient %s" % (
            variant, effects, patient.id)
        effect = effects[0]
        peptide_start_in_protein = row["peptide_start_in_protein"]
        peptide = row["peptide"]
        length = row["length"]
        slice_mutant = effect.mutant_protein_sequence[
            peptide_start_in_protein:peptide_start_in_protein + length]
        assert slice_mutant == peptide, (
            "Mutant protein sequence slice should equal peptide, but mutant slice = %s and peptide = %s" % (
                slice_mutant, peptide))
        peptide_wt = effect.original_protein_sequence[
            peptide_start_in_protein:peptide_start_in_protein + length]
        assert len(peptide_wt) == len(peptide), (
            "Wildtype sequence must be the same length as mutant sequence, but "
            "mutant length = %d and wildtype length = %d" % (
                len(peptide), len(peptide_wt)))
        return peptide_wt

    def normalize_score(row):
        # Trimmed length
        return row["score"] / (len(row["epitope"]) - 3)

    def normalize_score_wt(row):
        # Trimmed length
        return row["score_wt"] / (len(row["epitope_wt"]) - 3)

    selected_neoantigen_cols = ["patient_id", "peptide"]
    epitope_column_names = ["sample", "epitope"]
    compare_kwargs = {"epitope_lengths": cohort.epitope_lengths,
                      "include_hla": True,
                      "data_filters": [DataFilter(on="tuberculosis")],
                      "iedb_path": path.join(REPO_DATA_DIR, "iedb_tcell_data_6_10_15.csv")}

    if include_wildtype:
        df_neoantigens["peptide_wt"] = df_neoantigens.apply(wildtype_peptide, axis=1)
        selected_neoantigen_cols.append("peptide_wt")
        epitope_column_names.append("epitope_wt")
        compare_kwargs["include_wildtype"] = True

    if include_organism:
        compare_kwargs["include_organism"] = True

    df_epitopes = df_neoantigens[selected_neoantigen_cols]
    df_epitopes.columns = epitope_column_names

    compare_kwargs["epitopes"] = df_epitopes
    df_homology = compare(**compare_kwargs)
    df_homology.rename(columns={"hla": "iedb_hla_allele"}, inplace=True)

    if include_wildtype:
        df_homology["score_wt_normalized"] = df_homology.apply(normalize_score_wt, axis=1)

    df_homology["score_normalized"] = df_homology.apply(normalize_score, axis=1)

    # Add "allele" back in
    df_homology = df_homology.merge(
        df_neoantigens[["patient_id", "peptide", "allele"]],
        left_on=["sample", "epitope"],
        right_on=["patient_id", "peptide"])
    df_homology.rename(columns={"allele": "peptide_hla_allele"}, inplace=True)
    df_homology.peptide_hla_allele = df_homology.peptide_hla_allele.apply(compact_allele_name)

    cohort.save_to_cache(df_homology, cohort.cache_names["homology"], patient.id, cached_file_name)
    return df_homology

def load_iedb_binders(cohort):
    all_alleles = []
    for patient in cohort:
        all_alleles.extend(patient.hla_alleles)
    all_alleles = list(set(all_alleles))
    print("All alleles: %s (%d)" % (all_alleles, len(all_alleles)))

    df_iedb = get_iedb_epitopes(
        epitope_lengths=cohort.epitope_lengths,
        data_filters=[DataFilter(on="tuberculosis")],
        iedb_path=path.join(REPO_DATA_DIR, "iedb_tcell_data_6_10_15.csv"))

    dfs = []
    for allele in all_alleles:
        df_iedb_binders = cohort.load_single_allele_iedb_binders(allele, df_iedb=df_iedb)
        dfs.append(df_iedb_binders)
    return pd.concat(dfs)

def load_single_allele_iedb_binders(cohort, allele, df_iedb=None):
    # This is a hack; using alleles as directories rather than patients
    cached_file_name = "%s-%s-iedb-binders.csv" % (cohort.variant_type, cohort.merge_type)
    df_iedb_binders = cohort.load_from_cache(cohort.cache_names["iedb-binders"], allele, cached_file_name)
    if df_iedb_binders is not None:
        return df_iedb_binders

    if df_iedb is None:
        df_iedb = get_iedb_epitopes(
            epitope_lengths=cohort.epitope_lengths,
            data_filters=[DataFilter(on="tuberculosis")],
            iedb_path=path.join(REPO_DATA_DIR, "iedb_tcell_data_6_10_15.csv"))

    protein_sequences = dict(zip(list(df_iedb.iedb_epitope), list(df_iedb.iedb_epitope)))
    print("Predicting binding for %d IEDB sequences" % len(protein_sequences))
    mhc_model = cohort.mhc_class(
        alleles=[allele],
        epitope_lengths=cohort.epitope_lengths,
        max_file_records=None,
        process_limit=30)
    df_iedb_binders = mhc_model.predict(protein_sequences).to_dataframe()

    # This is a hack; using alleles as directories rather than patients
    cohort.save_to_cache(df_iedb_binders, cohort.cache_names["iedb-binders"], allele, cached_file_name)
    return df_iedb_binders

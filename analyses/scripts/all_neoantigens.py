import sys
from os import getcwd, path
sys.path.append(path.dirname(getcwd()))
import matplotlib.pyplot as plt
import seaborn as sb

from utils import data

from cohorts.functions import neoantigen_count, missense_snv_count

cohort = data.init_cohort(cache_data_dir="/home/tavi/melanoma-reanalysis/data/cache")

df = cohort.as_dataframe(on=[neoantigen_count, missense_snv_count])

effects = cohort.load_effects()

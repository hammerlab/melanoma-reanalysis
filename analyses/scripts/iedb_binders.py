import sys
from os import getcwd, path
sys.path.append(path.dirname(getcwd()))
import matplotlib.pyplot as plt
import seaborn as sb

from utils import data
from utils.functions import *

cohort = data.init_cohort(cache_data_dir="/home/tavi/melanoma-reanalysis/data/cache",
                          four_callers=True,
                          biopsy_time="pre",
                          non_discordant=True)

df = cohort.load_iedb_binders()

import pathlib

BASE = pathlib.Path.cwd().parent

LIB = BASE / 'lib'

DATA = BASE / 'data'
DRUGS = DATA / 'drugs'
TARGETS = DATA / 'targets'
EMBEDS = TARGETS / 'embeds'
KIBA = DATA / 'kiba'
TEMP = DATA / 'temp'

EXPERIMENTS = BASE / 'experiments'
MODELS = EXPERIMENTS / 'models'
PLOTS = EXPERIMENTS / 'plots'

XLSX_PATH = KIBA / 'kiba.xlsx'
FASTA_PATH = TARGETS / 'target_sequence.fasta'

import numpy as np
import pandas as pd
import requests

from lib.dataset.affinity import AffinityData
from lib.dataset.minimal_dta import MinimalDTA
from lib.util.anchor_util import KIBA, XLSX_PATH
from lib.util.log_util import log_func
from lib.util.load_util import load


@log_func
def download_kiba():
    url = "https://ndownloader.figstatic.com/files/3950161"
    response = requests.get(url)
    with open(XLSX_PATH, 'wb') as f:
        f.write(response.content)


@log_func
@load(KIBA, 'kiba.pandas')
def load_kiba_pandas():
    if not XLSX_PATH.exists():
        download_kiba()
    return pd.read_excel(
        XLSX_PATH,
        sheet_name="KIBA",
        header=0,
        index_col=0
    )


THRESHOLD = 3.0


@log_func
@load(KIBA, 'kiba.pickle')
def load_kiba() -> MinimalDTA:
    df = load_kiba_pandas()
    target_ids = df.columns.values.tolist()
    drug_ids = []
    affinities = {}
    for i, row in enumerate(df.itertuples(index=True, name=None)):
        current_drug = row[0]
        for j, cell in enumerate(row[1:]):
            if not np.isnan(cell):
                affinities[i, j] = float(cell) < THRESHOLD
        drug_ids.append(current_drug)

    return MinimalDTA(
        name='kiba',
        drug_ids=dict(enumerate(drug_ids)),
        target_ids=dict(enumerate(target_ids)),
        affinities=AffinityData('kiba_affinities', affinities),
    )

from typing import Any, Callable

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, Lipinski, Crippen

from lib.dataset.representation import RepresentationData
from lib.load.minimal_dta import load_kiba
from lib.util.anchor_util import TEMP, DRUGS
from lib.util.load_util import load
from lib.util.log_util import log_func, logger
from lib.util.representation_util import batch_generated, romol_representation, safe_generate


def load_drug_ids() -> dict[int, Any]:
    return load_kiba().drug_ids


@batch_generated(load_drug_ids, DRUGS, TEMP, 'drug_smiles')
def load_drug_smiles(batch_dict):
    id_list = list(batch_dict.values())

    drug_provider = new_client.molecule \
        .filter(molecule_chembl_id__in=id_list) \
        .only('molecule_chembl_id', 'molecule_structures')
    drug_records = list(drug_provider)
    drug_df = pd.DataFrame.from_records(drug_records, index='molecule_chembl_id')

    result = {}

    for i, drug_id in batch_dict.items():
        try:
            result[i] = drug_df.loc[drug_id, 'molecule_structures']['canonical_smiles']
        except (KeyError, TypeError):
            logger.warning(f'No available drug_smiles.pickle of {drug_id : >20}')

    return result


@log_func
@load(DRUGS, 'drug_romol.pickle')
def load_drug_romol():
    return safe_generate(
        name='romol',
        items=load_drug_smiles(),
        generator=Chem.MolFromSmiles,
        name_map=load_drug_ids()
    )


##################################################

drug_repr_loaders = []


def _drug_repr_loader(func: Callable[[], RepresentationData]):
    drug_repr_loaders.append(func)
    return func


# @_drug_repr_loader
@romol_representation(DRUGS, 'drug_basic_attributes', load_drug_ids, load_drug_romol)
def load_drug_basic_attributes(molecule):
    return np.array([
        Descriptors.ExactMolWt(molecule),
        Lipinski.NumHAcceptors(molecule),
        Lipinski.NumHDonors(molecule),
        Crippen.MolLogP(molecule)]
    )


@_drug_repr_loader
@romol_representation(DRUGS, 'drug_maccs', load_drug_ids, load_drug_romol)
def load_drug_maccs(molecule):
    return np.array(MACCSkeys.GenMACCSKeys(molecule).ToList())


@_drug_repr_loader
@romol_representation(DRUGS, 'drug_rdkit', load_drug_ids, load_drug_romol)
def load_drug_rdkit(molecule):
    return np.array(Chem.RDKFingerprint(molecule, fpSize=2048).ToList())


@_drug_repr_loader
@romol_representation(DRUGS, 'drug_morgan', load_drug_ids, load_drug_romol)
def load_drug_morgan(molecule):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2).ToList())



import logging
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import polars as pl
from matchms.importing import load_from_mgf
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def convert_mgf_ipc(
    source: Path,
) -> pl.DataFrame:
    """Convert .mgf file to Polars .ipc."""
    schema = {
        "experiment_name": str,
        "index": int,
        "sequence": str,
        "modified_sequence": str,
        "precursor_mz": pl.Float64,
        "precursor_charge": int,
        "mz_array": pl.List(pl.Float64),
        "intensity_array": pl.List(pl.Float64),
    }

    df = pl.DataFrame(schema=schema)


    index = 1
    filepath = source
    exp = load_from_mgf(str(filepath))

    data = []

    for spectrum in tqdm(exp):
        meta = spectrum.metadata
        peptide = ""
        unmod_peptide = ""
        if "peptide_sequence" in meta:
            peptide = meta["peptide_sequence"]
            unmod_peptide = "".join([x[0] for x in re.split(r"(?<=.)(?=[A-Z])", peptide)])
        if "charge" not in meta:
            print(f"Charge not found for {meta['peptide_sequence']}, skipping...")
            continue

        data.append(
            [
                filepath.parent.name+'_'+filepath.stem,
                index,
                unmod_peptide,
                peptide ,
                meta["precursor_mz"],
                meta["charge"],
                list(spectrum.mz),
                list(spectrum.intensities),
            ]
        )
        index += 1
    data_df = pl.DataFrame(data, schema=schema)

    df = pl.concat([df, data_df], how="diagonal")
    return df
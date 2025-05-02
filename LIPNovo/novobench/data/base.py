from typing import Dict, Sequence, Optional
import polars as pl
import numpy as np

class SpectrumData:
    """Base data object for spectrum data wrapped around a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame, annotated: bool = True):
        """
        Initializes the SpectrumData object.

        Parameters:
        - df (pl.DataFrame): A Polars DataFrame containing the spectrum data.
        - annotated (bool): A flag indicating whether the data is annotated with peptide sequences.
        """
        self._df = df
        self.annotated = annotated

    @property
    def df(self):
        return self._df

    def set_df(self, df):
        self._df = df

    @property
    def precursor_mz(self):
        return self.df.get_column('precursor_mz')

    @property
    def precursor_charge(self):
        return self.df.get_column('precursor_charge')

    @property
    def mz_array(self):
        return self.df.get_column('mz_array')

    @property
    def intensity_array(self):
        return self.df.get_column('intensity_array')

    @property
    def modified_sequence(self):
        return self.df.get_column('modified_sequence')


class SpectrumDataManager:
    def __init__(self, data: Dict[str, SpectrumData]):
          if not isinstance(data, dict):
            raise TypeError("data should be a dictionary of SpectrumData.")
          self._data = data

    @property
    def data(self):
        return self._data

    def get_train(self) -> Optional[SpectrumData]:
        return self._data.get("train") if "train" in self._data else None

    def get_valid(self) -> Optional[SpectrumData]:
        return self._data.get("valid") if "valid" in self._data else None

    # def get_test(self) -> Optional[SpectrumData]:
    #     return self._data.get("test") if "test" in self._data else None

      






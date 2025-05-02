import torch
from torch.utils.data import Dataset, DataLoader
from novobench.data import SpectrumData
import polars as pl
import numpy as np
from typing import List, Tuple, Optional  # Added missing imports
import re


PROTON_MASS_AMU = 1.007276466812  # Adjust as needed

class ImpnovoDataset(Dataset):
    """A Dataset to handle spectrum data stored in a Polars DataFrame."""

    def __init__(self, data: SpectrumData):
        """
        Initializes the dataset with a preprocessed Polars DataFrame.

        Parameters:
        ----------
        data : SpectrumData
            the spectrum data.
        """
        super().__init__()
        self.df = data.df

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int, torch.Tensor | str]:
        mz_array = torch.tensor(self.df[idx, "mz_array"].to_list(), dtype=torch.float32)
        intensity_array = torch.tensor(self.df[idx, "intensity_array"].to_list(), dtype=torch.float32)
        precursor_mz = self.df[idx, "precursor_mz"]
        precursor_charge = self.df[idx, "precursor_charge"]

        peptide = ''
        if 'modified_sequence' in self.df.columns:
            peptide = self.df[idx, 'modified_sequence'] 
        else:
            peptide = self.df[idx, 'sequence'] 



        spectrum = torch.stack([mz_array, intensity_array], dim=1)

        return spectrum, precursor_mz, precursor_charge, peptide


class ImpnovoDataModule:
    """
    A simplified data loader for the de novo sequencing task.

    Parameters:
    ----------
    dataframe : pl.DataFrame
        The DataFrame containing the spectrum data.
    batch_size : int
        The batch size to use.
    n_workers : int, optional
        The number of workers to use for data loading. By default, it uses 0 (main process).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        batch_size: int = 128,
        n_workers: Optional[int] = 0,
    ):
        self.dataframe = df
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dataset = ImpnovoDataset(df)

    def get_dataloader(self,shuffle=False) -> DataLoader:
        """
        Create and return a PyTorch DataLoader.

        Returns:
        -------
        DataLoader: A PyTorch DataLoader for the spectrum data.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            shuffle = shuffle
        )

def collate_batch(batch: List[Tuple[torch.Tensor, float, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Collate MS/MS spectra into a batch, similar to the prepare_batch function.

    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, float, int, str]]
        A batch of data consisting of for each spectrum (i) a tensor with the m/z and intensity peak values,
        (ii) the precursor m/z, (iii) the precursor charge, (iv) the spectrum identifier or peptide sequence.

    Returns
    -------
    spectra : torch.Tensor
        The padded mass spectra tensor with the m/z and intensity peak values for each spectrum.
    precursors : torch.Tensor
        A tensor with the precursor neutral mass, precursor charge, and precursor m/z.
    spectrum_ids : np.ndarray
        An array of spectrum identifiers or peptide sequences.
    """
    spectra, precursor_mzs, precursor_charges, spectrum_ids = zip(*batch)

    
    # Pad spectra to create a uniform tensor
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)

    
    # Convert precursor information to tensors
    precursor_mzs = torch.tensor(precursor_mzs, dtype=torch.float32)
    precursor_charges = torch.tensor(precursor_charges, dtype=torch.float32)
    
    # Calculate precursor masses and create a tensor for precursor data
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.stack([precursor_masses, precursor_charges, precursor_mzs], dim=1)
    
    # Convert spectrum identifiers or peptide sequences to a NumPy array
    spectrum_ids = np.array(spectrum_ids)

    return spectra, precursors, spectrum_ids

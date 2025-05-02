import os
import pathlib
import polars as pl
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from novobench.transforms import BaseTransform
from novobench.data import SpectrumData, SpectrumDataManager

class BaseDataset(ABC):
    """
    An abstract class representing a dataset.

    Attributes:
        root (pathlib.Path): The root directory where the dataset is stored.

    Methods:
        _load_raw_data: Abstract method to load raw data from the dataset.
        _raw_to_pynovo: Abstract method to convert raw data to PyNovo data objects.
        load_data: Loads the dataset, optionally applies a transformation, and returns PyNovo data objects.
    """
    
    def __init__(self, root: str):
        """Initializes the dataset with the given root directory."""
        self.root = pathlib.Path(root).resolve()
        print(f"self.root = {self.root}")

    @abstractmethod
    def _load_raw_data(self) -> Any:
        """Abstract method to load raw data from the dataset."""
        pass

    @abstractmethod
    def _raw_to_pynovo(self, raw_data: Any) -> SpectrumDataManager:
        """
        Abstract method to convert raw data into PyNovo data objects.

        Parameters:
            raw_data (Any): The raw data to be converted.

        Returns:
            SpectrumDataModule: The converted PyNovo data object.
        """
        pass

    def load_data(self, transform: Optional[BaseTransform] = None) -> SpectrumDataManager:
        """
        Loads the dataset, applies an optional transformation, and returns PyNovo data objects.

        Parameters:
            transform (Optional[BaseTransform]): An optional transformation to apply to the data.

        Returns:
            SpectrumData: The loaded and optionally transformed PyNovo data object.

        Raises:
            TypeError: If the provided transform is not a BaseTransform instance.
        """
        raw_data = self._load_raw_data()
        pynovo_data = self._raw_to_pynovo(raw_data)
        if not isinstance(pynovo_data.data, dict):
            raise TypeError("pynovo_data.data should be a dictionary of SpectrumData.")

        if transform is not None:
            if not isinstance(transform, BaseTransform):
                raise TypeError(f"Transform must be an instance of BaseTransform, got {type(transform)} instead.")

            for split, spectrum in pynovo_data.data.items():
                print(f"Applying transform to {split} split.")
                pynovo_data.data[split] = transform(spectrum)

        return pynovo_data


    
class CustomDataset(BaseDataset):
    """
    A custom dataset class inheriting from BaseDataset for handling specific dataset formats.

    Attributes:
        files (Dict[str, str]): A dictionary mapping dataset names to their file paths.

    Methods:
        _load_raw_data: Implementation of the abstract method to load raw data specific to this dataset.
        _raw_to_pynovo: Implementation of the abstract method to convert raw data to PyNovo data objects.
    """

    def __init__(self, data_dir: str, files: Dict[str, str]):
        """
        Initializes the custom dataset with the given directory and file mappings.

        Parameters:
            data_dir (str): The directory where the dataset files are located.
            files (Dict[str, str]): A dictionary mapping dataset names to their file paths.
        """
        super().__init__(data_dir)
        self.files = files
        print(f"self.files = {files}")

    def _load_raw_data(self) -> Dict[str, pl.DataFrame]:
        """
        Loads raw data from the specified files in the dataset directory.

        Returns:
            Dict[str, pl.DataFrame]: A dictionary of DataFrames loaded from the dataset files.
        """
        df_dict = {}
        for df_name, file_path in self.files.items():
            df_path = self.root / file_path
            if df_path.exists():
                df_dict[df_name] = pl.read_parquet(df_path)
                print(f"Loaded {df_name} from {df_path}.")
            else:
                raise FileNotFoundError(f"File {df_path} not found.")

        return df_dict

    def _raw_to_pynovo(self, raw_data: Dict[str, pl.DataFrame]) -> SpectrumDataManager:
        """
        Converts the loaded raw data into PyNovo data objects.

        Parameters:
            raw_data (Dict[str, pl.DataFrame]): The raw data loaded from the dataset files.

        Returns:
            SpectrumDataModule: The converted PyNovo data object.
        """
        spectrum_data_dict = {}
        for split, df in raw_data.items():
            spectrum_data_dict[split] = SpectrumData(df)
        return SpectrumDataManager(spectrum_data_dict)

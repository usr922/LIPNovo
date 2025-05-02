import shutil
from novobench.data import SpectrumDataManager, SpectrumData
from .base import BaseDataset
import pathlib
import polars as pl
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

class NineSpeciesDataset(BaseDataset):
    def __init__(self, data_dir: str, task: str):
        super().__init__(data_dir)

        self.task = task
        self.file_mapping = {
            "cross9.exclude_honeybee": 
            {
                "train" : "example_honeybee.parquet",
                "valid" : "example_honeybee.parquet",
                "test" : "example_honeybee.parquet"
            }
        }
        if task in self.file_mapping:
            self.files = self.file_mapping[task]
        else:
            raise ValueError(f"Unrecognized task name: {task}")


    def check_download(self):
        """Download selected files of the dataset."""
        ...


    def _load_raw_data(self) -> Any:
        ...


    def _raw_to_pynovo(self, raw_data: Any, /) -> SpectrumDataManager:
        ...







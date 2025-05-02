"""
Training and testing functionality for the de novo peptide sequencing model.
"""

import glob
import logging
import os
import sys
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union
from pathlib import Path
import lightning.pytorch as pln
import polars as pl
import numpy as np
import torch
import time

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything

from novobench.data import ms_io
from .impnovo_dataloader import ImpnovoDataset, ImpnovoDataModule
from .impnovo_modeling import Spec2Pep

from novobench.transforms import SetRangeMZ, FilterIntensity, RemovePrecursorPeak, ScaleIntensity
from novobench.transforms.misc import Compose
from novobench.utils.preprocessing import convert_mgf_ipc
from novobench.data import SpectrumData
logger = logging.getLogger("impnovo")



def init_logger(config):
    # Set up logging
    output = config.logger_save_path
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(output)
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    

class ImpnovoRunner:
    """A class to run Impnovo models.
    Parameters
    ----------
    config : Config object
        The impnovo configuration.
    model_filename : str, optional
        The model filename is required for eval and de novo modes,
        but not for training a model from scratch.
    """

    def __init__(
        self,
        config,
        model_filename: Optional[str] = None,
        saved_path: Optional[str] = "",
    ) -> None:
        
        seed_everything(seed=config.random_seed, workers=True)
        init_logger(config)
        """Initialize a ModelRunner"""
        self.config = config
        self.model_filename = model_filename
        self.saved_path = saved_path

        # Initialized later:
        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None

        # Configure checkpoints.
        if config.save_top_k is not None:
            self.callbacks = [
                ModelCheckpoint(
                    dirpath=config.model_save_folder_path,
                    monitor="valid_CELoss",
                    mode="min",
                    save_top_k=config.save_top_k,
                )
            ]
        else:
            self.callbacks = None


    @staticmethod
    def preprocessing_pipeline(config):
        transforms = [
            SetRangeMZ(config.min_mz, config.max_mz), 
            RemovePrecursorPeak(config.remove_precursor_tol),
            FilterIntensity(config.min_intensity, config.n_peaks),
            ScaleIntensity()
        ]
        return Compose(*transforms)
    

    def __enter__(self):
        """Enter the context manager"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup on exit"""
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
    ) -> None:
        """Train the Impnovo model.

        Parameters
        ----------
        train_peak_path : iterable of str
            The path to the MS data files for training.
        valid_peak_path : iterable of str
            The path to the MS data files for validation.

        Returns
        -------
        self
        """
        self.initialize_trainer(train=True)
        self.initialize_model(train=True)
        
        train_loader = ImpnovoDataModule(
            df = train_df,
            n_workers=self.config.n_workers,
            batch_size=self.config.train_batch_size // self.trainer.num_devices
        ).get_dataloader(shuffle=True)
        
        val_loader = ImpnovoDataModule(
            df = val_df,
            n_workers=self.config.n_workers,
            batch_size=self.config.train_batch_size // self.trainer.num_devices
        ).get_dataloader()

        start_time = time.time()
        self.trainer.fit(
            self.model,
            train_loader,
            val_loader,
        )
        training_time = time.time() - start_time
        logger.info(f"Training took {training_time:.2f} seconds")

    def denovo(self, test_df: pl.DataFrame,) -> None:
        """Evaluate peptide sequence preditions from a trained Impnovo model.

        Parameters
        ----------
        peak_path : iterable of str
            The path with MS data files for predicting peptide sequences.

        Returns
        -------
        self
        """
        self.initialize_trainer(train=False)
        self.initialize_model(train=False)
        test_loader = ImpnovoDataModule(
            df = test_df,
            n_workers=self.config.n_workers,
            batch_size=self.config.predict_batch_size // self.trainer.num_devices 
        ).get_dataloader()
        
        start_time = time.time()
        self.trainer.validate(self.model, test_loader)
        training_time = time.time() - start_time
        logger.info(f"denovo took {training_time:.2f} seconds")



    def initialize_trainer(self, train: bool) -> None:
        """Initialize the lightning Trainer.

        Parameters
        ----------
        train : bool
            Determines whether to set the trainer up for model training
            or evaluation / inference.
        """
        trainer_cfg = dict(
            accelerator = self.config.accelerator,
            enable_checkpointing=False,
        )

        if train:
            if self.config.devices is None:
                devices = "auto"
            else:
                devices = self.config.devices
            
            additional_cfg = dict(
                devices=devices,
                callbacks=self.callbacks,
                enable_checkpointing=self.config.save_top_k is not None,
                max_epochs=self.config.max_epochs,
                num_sanity_val_steps=self.config.num_sanity_val_steps,
                strategy=self._get_strategy(),
                val_check_interval=self.config.val_check_interval,
                check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            )
            trainer_cfg.update(additional_cfg)

        self.trainer = pln.Trainer(**trainer_cfg)

    def initialize_model(self, train: bool) -> None:
        """Initialize the Impnovo model.

        Parameters
        ----------
        train : bool
            Determines whether to set the model up for model training
            or evaluation / inference.
        """
        model_params = dict(
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            dim_intensity=self.config.dim_intensity,
            max_length=self.config.max_length,
            residues=self.config.residues,
            max_charge=self.config.max_charge,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            min_peptide_len=self.config.min_peptide_len,
            n_beams=self.config.n_beams,
            top_match=self.config.top_match,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            max_iters=self.config.max_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gen_num=self.config.gen_num,
            gen_enc_layers=self.config.gen_enc_layers,
            gen_dec_layers=self.config.gen_dec_layers,
            gen_threshold=self.config.gen_threshold,
        )

        # Reconfigurable non-architecture related parameters for a loaded model
        loaded_model_params = dict(
            max_length=self.config.max_length,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            n_beams=self.config.n_beams,
            min_peptide_len=self.config.min_peptide_len,
            top_match=self.config.top_match,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            max_iters=self.config.max_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gen_num=self.config.gen_num,
            gen_enc_layers=self.config.gen_enc_layers,
            gen_dec_layers=self.config.gen_dec_layers,
            gen_threshold=self.config.gen_threshold,
        )

        if self.model_filename is None:
            # Train a model from scratch if no model file is provided.
            if train:
                self.model = Spec2Pep(**model_params)
                return
            # Else we're not training, so a model file must be provided.
            else:
                logger.error("A model file must be provided")
                raise ValueError("A model file must be provided")
        # Else a model file is provided (to continue training or for inference).

        if not Path(self.model_filename).exists():
            logger.error(
                "Could not find the model weights at file %s",
                self.model_filename,
            )
            raise FileNotFoundError("Could not find the model weights file")

        # First try loading model details from the weights file, otherwise use
        # the provided configuration.
        device = torch.empty(1).device  # Use the default device.
        try:
            self.model = Spec2Pep.load_from_checkpoint(
                self.model_filename, map_location=device, saved_path=self.saved_path, **loaded_model_params
            )

            architecture_params = set(model_params.keys()) - set(
                loaded_model_params.keys()
            )
            for param in architecture_params:
                if model_params[param] != self.model.hparams[param]:
                    warnings.warn(
                        f"Mismatching {param} parameter in "
                        f"model checkpoint ({self.model.hparams[param]}) "
                        f"vs config file ({model_params[param]}); "
                        "using the checkpoint."
                    )
        except RuntimeError:
            # This only doesn't work if the weights are from an older version
            try:
                self.model = Spec2Pep.load_from_checkpoint(
                    self.model_filename,
                    map_location=device,
                    **model_params,
                )
            except RuntimeError:
                raise RuntimeError(
                    "Weights file incompatible with the current version of "
                    "Impnovo. "
                )


    def _get_strategy(self) -> Union[str, DDPStrategy]:
        """Get the strategy for the Trainer.

        The DDP strategy works best when multiple GPUs are used. It can work
        for CPU-only, but definitely fails using MPS (the Apple Silicon chip)
        due to Gloo.

        Returns
        -------
        Union[str, DDPStrategy]
            The strategy parameter for the Trainer.

        """
        if self.config.accelerator in ("cpu", "mps"):
            return "auto"
        elif self.config.devices == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"



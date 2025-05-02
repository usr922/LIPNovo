"""
Extended from https://github.com/Noble-Lab/casanovo
Reference
----------
Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo mass spectrometry peptide sequencing with a transformer model. 
in Proceedings of the 39th International Conference on Machine Learning - ICML '22 vol. 162 25514–25522 (PMLR, 2022). https://proceedings.mlr.press/v162/yilmaz22a.html

"""

import collections
import heapq
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import depthcharge
import einops
import torch
import torch.nn as nn
import numpy as np
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter
from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder
import torch.nn.functional as F

from novobench.metrics import evaluate
import pandas as pd

logger = logging.getLogger("impnovo")



class MLP(pl.LightningModule, ModelMixin):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class Imputator(pl.LightningModule, ModelMixin):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        n_enc_layers=3,
        n_dec_layers=3
        ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_enc_layers
        ) if n_enc_layers > 0 else None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout)

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_dec_layers
        )

        self.pred_head = MLP(
            input_dim=dim_model, 
            hidden_dim=dim_model*4, 
            output_dim=dim_model,
            num_layers=3)
        
        self.cls_head = MLP(
            input_dim=dim_model, 
            hidden_dim=dim_model*4, 
            output_dim=1,
            num_layers=3)
    
    def forward(self, src, masks, query_embed):
        """
        memories: extracted memories from exp peaks
        masks: masks
        """
        bs, l, dim = src.shape

        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        memories = self.encoder(src, src_key_padding_mask=masks) if self.encoder is not None else src

        out = self.decoder(
            tgt=query_embed,
            memory=memories,
            memory_key_padding_mask=masks,
        )

        pred = self.pred_head(out)
        cls_logit = self.cls_head(out)

        return pred, cls_logit



class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    max_length : int
        The maximum peptide length to decode.
    residues: Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams: int
        Number of beams used during beam search decoding.
    top_match: int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter: Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    train_label_smoothing: float
        Smoothing factor when calculating the training loss.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    out_writer: Optional[str]
        The output writer for the prediction results.
    calculate_precision: bool
        Calculate the validation set precision during training.
        This is expensive.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 1,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter
        ] = None,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        out_writer= None,
        saved_path: str = "",
        gen_num = 50,
        gen_enc_layers=3,
        gen_dec_layers=3,
        gen_threshold=0.8,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gen_num = gen_num
        self.gen_enc_layers = gen_enc_layers
        self.gen_dec_layers = gen_dec_layers
        self.gen_threshold = gen_threshold

        self.saved_path = saved_path
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
        )

        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
        )

        num_queries = self.gen_num
        self.query_embed = nn.Embedding(num_queries, dim_model)
        self.num_queries = num_queries

        self.imputator = Imputator(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_enc_layers=self.gen_enc_layers,
            n_dec_layers=self.gen_dec_layers
        )

        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        self.gen_mseloss = torch.nn.MSELoss()
        self.gen_bceloss = torch.nn.BCEWithLogitsLoss()

        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues
        )
        self.stop_token = self.decoder._aa2idx["$"]

        # Logging.
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = None   # modified by sizhe for testing



    def hungarian_matching(self, AA, logits, BB, B_mask):
        from scipy.optimize import linear_sum_assignment

        def mse_distance(A, B):
            diff = A.unsqueeze(1) - B.unsqueeze(0)
            mse = torch.mean(diff ** 2, dim=-1)
            return mse
        
        cls_score = 1 - logits.squeeze(-1).sigmoid()

        bs = AA.shape[0]
        col_inds = []
        row_inds = []
        for b in range(bs):
            A = AA[b]
            B = BB[b]
            B = B[B_mask[b] == 0]
            cls_s = cls_score[b]
            mse_cost = mse_distance(A, B)
            cls_cost = cls_s.unsqueeze(-1)
            cost_matrix = mse_cost + cls_cost
            cost_matrix = cost_matrix.cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            row_inds.append(row_ind)
            col_inds.append(col_ind)

        return row_inds, col_inds


    def forward_gen(self, memories, src_key_padding_mask):

        bs = memories.shape[0]
        m_len = memories.shape[1]
        dim = memories.shape[-1]
        pred, cls_logit = self.imputator(memories, src_key_padding_mask, self.query_embed.weight)

        return pred, cls_logit

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        return self.beam_search_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )

    def beam_search_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        
        spectra = torch.cat([spectra, spectra], dim=1)
        spectra[:, (spectra.shape[1]//2):, 0] = spectra[:,(spectra.shape[1]//2):,0] - precursors[:,[0]]
        memories, mem_masks = self.encoder(spectra)        
        generated_embeddings, cls_logits = self.forward_gen(memories[:, 1:], mem_masks[:, 1:])
        gen_mask = cls_logits.squeeze(-1).sigmoid() < self.gen_threshold

        memories = torch.cat([
            memories[:, :1, :],
            generated_embeddings,
            memories[:, 1:, :]
        ], dim=1)

        mem_masks = torch.cat([
            mem_masks[:, :1],
            gen_mask,
            mem_masks[:, 1:],
        ], dim=1)

        # Sizes.
        batch = spectra.shape[0]  # B
        length = self.max_length + 1  # L
        vocab = self.decoder.vocab_size + 1  # V
        beam = self.n_beams  # S
        # Initialize scores and tokens.
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        )
        scores = scores.type_as(spectra)
        tokens = torch.zeros(batch, length, beam, dtype=torch.int64)
        tokens = tokens.to(self.encoder.device)

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        pred, _ = self.decoder(None, precursors, memories, mem_masks)
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        # The main decoding loop.
        for step in range(0, self.max_length):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
            ) = self._finish_beams(tokens, precursors, step)
            # Cache peptide predictions from the finished beams (but not the
            # discarded beams).
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )

            # Stop decoding when all current beams have been finished.
            # Continue with beams that have not been finished and not discarded.
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            # Update the scores.
            scores[~finished_beams, : step + 2, :], _ = self.decoder(
                tokens[~finished_beams, : step + 1],
                precursors[~finished_beams, :],
                memories[~finished_beams, :, :],
                mem_masks[~finished_beams, :],
            )
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return list(self._get_top_peptide(pred_cache))

    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """
        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass = [None]
        for aa, mass in self.peptide_mass_calculator.masses.items():
            if mass < 0:
                aa_neg_mass.append(aa)
        # Find N-terminal residues.
        n_term = torch.Tensor(
            [
                self.decoder._aa2idx[aa]
                for aa in self.peptide_mass_calculator.masses
                if aa.startswith(("+", "-"))
            ]
        ).to(self.decoder.device)

        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.encoder.device)
        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        ends_stop_token = tokens[:, step] == self.stop_token
        finished_beams[ends_stop_token] = True
        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        discarded_beams[tokens[:, step] == 0] = True
        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            # Multiple N-terminal modifications.
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)
            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])
            internal_mods = torch.isin(
                torch.where(mask.to(self.encoder.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)
            peptide = self.decoder.detokenize(pred_tokens)
            # Omit stop token.
            if self.decoder.reverse and peptide[0] == "$":
                peptide = peptide[1:]
                peptide_len -= 1
            elif not self.decoder.reverse and peptide[-1] == "$":
                peptide = peptide[:-1]
                peptide_len -= 1
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 2]
            matches_precursor_mz = exceeds_precursor_mz = False
            for aa in [None] if finished_beams[i] else aa_neg_mass:
                if aa is None:
                    calc_peptide = peptide
                else:
                    calc_peptide = peptide.copy()
                    calc_peptide.append(aa)
                try:
                    calc_mz = self.peptide_mass_calculator.mass(
                        seq=calc_peptide, charge=precursor_charge
                    )
                    delta_mass_ppm = [
                        _calc_mass_error(
                            calc_mz,
                            precursor_mz,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            self.isotope_error_range[0],
                            self.isotope_error_range[1] + 1,
                        )
                    ]
                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.
            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, amino acid-level scores, and the predicted tokens is
            stored.
        """
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            pred_tokens = tokens[i][: step + 1]
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            # Don't cache this peptide if it was already predicted previously.
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            smx = self.softmax(scores[i : i + 1, : step + 1, :])
            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].tolist()
            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            if not has_stop_token:
                aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)
            # Calculate the updated amino acid-level and the peptide scores.
            aa_scores, peptide_score = _aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )
            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    np.random.random_sample(),
                    aa_scores,
                    torch.clone(pred_peptide),
                ),
            )

    def _get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.decoder.vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores[:, step, :, :], "B V S -> B (V S)"
        )

        # Find all still active beams by masking out terminated beams.
        active_mask = (
            ~finished_beams.reshape(batch, beam).repeat(1, vocab)
        ).float()
        # Mask out the index '0', i.e. padding token, by default.
        # FIXME: Set this to a very small, yet non-zero value, to only
        # get padding after stop token.
        active_mask[:, :beam] = 1e-8

        # Figure out the top K decodings.
        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, step, :] = torch.tensor(v_idx)
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return tokens, scores

    def _get_top_peptide(
        self,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each spectrum.

        Parameters
        ----------
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the peptide
            score, amino acid-level scores, and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores,
                        "".join(self.decoder.detokenize(pred_tokens)),
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """

        encoded = self.encoder(spectra)
        return self.decoder(sequences, precursors, *encoded), encoded

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """

        precursors = batch[1]
        spectra = batch[0]
        spectra = torch.cat([spectra, spectra], dim=1)
        spectra[:, (spectra.shape[1]//2):, 0] = spectra[:,(spectra.shape[1]//2):,0] - precursors[:,[0]]

        batch = (spectra, batch[1], batch[2])

        pep_seq = batch[-1]
        the_mz = []
        max_len = 0
        for seq in pep_seq:
            _the = self.calculate_all_ion_mz(seq.replace("(+57.02)", "").replace("M(+15.99)", "1").replace("N(+.98)", "2").replace("Q(+.98)", "3").replace("M(ox)", "4"))
            max_len = max(_the.shape[0], max_len)
            the_mz.append(_the)
        the_mz_pad = []
        for _the in the_mz:
            tmp = torch.zeros(max_len).cuda()
            tmp[:_the.shape[0]] = _the
            the_mz_pad.append(tmp.unsqueeze(0))
        the_mz = torch.cat(the_mz_pad, dim=0)
        the_intensity = torch.zeros_like(the_mz)
        exp_intensity = batch[0][:, :, 1]
        exp_mask = exp_intensity > 0

        the_intensity = torch.max(exp_intensity, dim=-1)[0].unsqueeze(-1)  # [bs, 1]
        the_intensity = the_intensity.repeat(1, the_mz.shape[-1])
        the_mask = the_mz > 0
        the_intensity = the_intensity * the_mask
        the_mz = the_mz.unsqueeze(-1)
        the_intensity = the_intensity.unsqueeze(-1)

        the_ms = torch.cat([the_mz, the_intensity], dim=-1)
        the_batch = (the_ms, batch[1], batch[2])


        exp_encoded = self.encoder(batch[0])
        exp_memories = exp_encoded[0][:, 1:]
        exp_src_key_padding_mask = exp_encoded[1][:, 1:]
        # Note that here we remove the first token, because it is the virtual global token


        generated_embeddings, cls_logits = self.forward_gen(exp_memories, exp_src_key_padding_mask)        
        gen_mask = cls_logits.squeeze(-1).sigmoid() < self.gen_threshold  # [bs, max_len=50]

        # decode with generated embeddings
        precursors = batch[1]
        sequences = batch[2]

        memories = torch.cat([
            exp_encoded[0][:, :1, :],
            generated_embeddings,
            exp_encoded[0][:, 1:, :]
        ], dim=1)

        src_key_padding_mask = torch.cat([
            exp_encoded[1][:, :1],
            gen_mask,
            exp_encoded[1][:, 1:],
        ], dim=1)

        complement_encoded = (memories, src_key_padding_mask)
        pred, truth = self.decoder(sequences, precursors, *complement_encoded)
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)

        if mode == "train":

            loss_ce = self.celoss(pred, truth.flatten())

            the_encoded = self.encoder(the_batch[0])
            target_encoded = the_encoded
            the_pred, the_truth = self.decoder(sequences, precursors, *the_encoded)
            the_pred = the_pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            loss_the = self.celoss(the_pred, the_truth.flatten())

            tgt_embeddings = target_encoded[0][:, 1:].detach()
            tgt_src_mask = target_encoded[1][:, 1:].detach()

            if tgt_embeddings.shape[1] > self.num_queries:
                tgt_embeddings = tgt_embeddings[:, :self.num_queries, :]
                tgt_src_mask = tgt_src_mask[:, :self.num_queries]
            
            row_inds, col_inds = self.hungarian_matching(generated_embeddings, cls_logits, tgt_embeddings, tgt_src_mask)  # list of indices

            loss_mse = 0.
            bs = tgt_embeddings.shape[0]

            for ind in range(len(row_inds)):
                _row = row_inds[ind]
                _col = col_inds[ind]
                cur_gen = generated_embeddings[ind]
                matched_gen = cur_gen[_row]
                cur_tgt = tgt_embeddings[ind][tgt_src_mask[ind] == 0]
                matched_tgt = cur_tgt[_col]
                loss_mse += self.gen_mseloss(matched_gen, matched_tgt)      
            loss_mse = loss_mse / bs

            cls_label = torch.zeros_like(cls_logits.squeeze(-1)).to(cls_logits.device)
            for ind in range(len(row_inds)):
                _row = row_inds[ind]
                cls_label[ind][_row] = 1
            loss_bce = self.gen_bceloss(cls_logits.squeeze(-1), cls_label)
            loss = loss_ce + loss_the + loss_bce + loss_mse

        else:
            loss = self.val_celoss(pred, truth.flatten())
            
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of  (i)     MS/MS spectra,[batch_size, peak_num, 2] 
                        (ii)    precursor information, [batch_size, 3],mass,charge,m/z
                        (iii)   peptide sequences.[batch_size],peptides

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # import pdb;pdb.set_trace()
        
        # Record the loss.
        loss = self.training_step(batch, mode="valid")

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.

        if not self.saved_path == "":
            peptides_pred, peptides_true, peptides_score = [], batch[2], []
            for spectrum_preds in self.forward(batch[0], batch[1]):
                if spectrum_preds == []:
                    peptides_pred.append("")
                    peptides_score.append(float('-inf'))
                for pep_score, _, pred in spectrum_preds:
                    peptides_pred.append(pred)
                    peptides_score.append(pep_score)
        
            assert(len(peptides_pred)==len(peptides_true) and len(peptides_score)==len(peptides_true))
            # Save the predicted peptides to a file.(denovo)
            batch_df = pd.DataFrame({
                'peptides_true': peptides_true,
                'peptides_pred': peptides_pred,
                'peptides_score': peptides_score
            })
            batch_df.to_csv(self.saved_path, mode='a', header=False, index=False)

        return loss


    def calculate_all_ion_mz(self, sequence):

        AA_MASS = {
                    "G": 57.021464, 
                    "A": 71.037114, 
                    "S": 87.032028, 
                    "P": 97.052764, 
                    "V": 99.068414, 
                    "T": 101.047670, 
                    "C": 160.030649, 
                    "L": 113.084064, 
                    "I": 113.084064, 
                    "N": 114.042927, 
                    "D": 115.026943, 
                    "Q": 128.058578, 
                    "K": 128.094963, 
                    "E": 129.042593, 
                    "M": 131.040485, 
                    "H": 137.058912, 
                    "F": 147.068414, 
                    "R": 156.101111, 
                    "Y": 163.063329,
                    "W": 186.079313, 
                    "1": 147.035400, 
                    "2": 115.026943, 
                    "3": 129.042594,
                    "4": 147.035400
                    }
        
        PROTON_MASS = 1.00728
        WATER_MASS = 18.01056
        AMMONIA_MASS = 17.02655
        OH_MASS = 17.0027
        H_MASS = 1.0078
        CO_MASS = 28.0100
        COOH_MASS = 45.018
        NH2_MASS = 16.0260
        
        n = len(sequence)

        alls = []
        b_ions = []
        y_ions = []
        
        for j in range(n-1):
            b_mass = 0
            for i in range(j+1):
                aa = sequence[i]
                b_mass += AA_MASS[aa]
                b_mass += WATER_MASS
            
            b_mass -= WATER_MASS * j
            b_mass -= OH_MASS
            b_ions.append(b_mass)
            
        rev_sequence = sequence[::-1]
        
        for j in range(n-1):
            y_mass = 0
            for i in range(j+1):
                aa = rev_sequence[i]
                y_mass += AA_MASS[aa]
                y_mass += WATER_MASS
            y_mass -= WATER_MASS * j
            y_mass -= H_MASS
            y_mass += H_MASS * 2
            y_ions.append(y_mass)

        alls = b_ions + y_ions 
        alls.sort()
        
        return torch.Tensor(alls)

  
    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["train_CELoss"].detach()
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss.item(),
        }
        self._history.append(metrics)
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "step": self.trainer.global_step,
            "valid": callback_metrics["valid_CELoss"].detach().item(),
        }
        self._history.append(metrics)
        self._log_history()


    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain loss\tValid loss\t"
            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f\t%.6f"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
                metrics.get("valid", np.nan),
            ]
            logger.info(msg % tuple(vals))


    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup_iters, max_iters=self.max_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


def _aa_pep_score(
    aa_scores: np.ndarray, fits_precursor_mz: bool
) -> Tuple[np.ndarray, float]:
    """
    Calculate amino acid and peptide-level confidence score from the raw amino
    acid scores.

    The peptide score is the mean of the raw amino acid scores. The amino acid
    scores are the mean of the raw amino acid scores and the peptide score.

    Parameters
    ----------
    aa_scores : np.ndarray
        Amino acid level confidence scores.
    fits_precursor_mz : bool
        Flag indicating whether the prediction fits the precursor m/z filter.

    Returns
    -------
    aa_scores : np.ndarray
        The amino acid scores.
    peptide_score : float
        The peptide score.
    """
    peptide_score = np.mean(aa_scores)
    aa_scores = (aa_scores + peptide_score) / 2
    if not fits_precursor_mz:
        peptide_score -= 1
    return aa_scores, peptide_score

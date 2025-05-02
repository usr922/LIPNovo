"""Methods to evaluate peptide-spectrum predictions."""
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import os,sys

from spectrum_utils.utils import mass_diff
from sklearn.metrics import auc
mass_Phosphorylation = 79.96633
STD_AA_MASS = {
    'G': 57.02146372057,
    'A': 71.03711378471,
    'S': 87.03202840427001,
    'P': 97.05276384885,
    'V': 99.06841391299,
    'T': 101.04767846841,
    'C': 103.00918478471,
    'L': 113.08406397713001,
    'I': 113.08406397713001,
    'J': 113.08406397713001,
    'N': 114.04292744114001,
    'D': 115.02694302383001,
    'Q': 128.05857750527997,
    'K': 128.09496301399997,
    'E': 129.04259308796998,
    'M': 131.04048491299,
    'H': 137.05891185845002,
    'F': 147.06841391298997,
    'U': 150.95363508471,
    'R': 156.10111102359997,
    'Y': 163.06332853254997,
    'W': 186.07931294985997,
    'O': 237.14772686284996,
    'M(+15.99)': 147.0354,
    'M(ox)': 147.0354,
    'Q(+.98)': 129.0426,
    'Q(Deamidation)': 129.0426,
    'N(+.98)': 115.02695,
    'N(Deamidation)': 115.02695,
    'C(+57.02)': 160.03065,
    'S(Phosphorylation)': 87.03203 + mass_Phosphorylation,
    'T(Phosphorylation)': 101.04768 + mass_Phosphorylation,
    'Y(Phosphorylation)': 163.06333 + mass_Phosphorylation,
}

def split_peptide(peptide, aa_dict):
    aa_list = aa_dict.keys()
    regex_pattern = '|'.join(map(re.escape, sorted(aa_list, key=len, reverse=True)))


    parts = re.findall(regex_pattern, peptide)

    parts = [part for part in parts if part]

    if ''.join(parts) != peptide:
        raise ValueError(f"Input string '{peptide}' contains unmatched characters in aa_dict: {aa_list}.")
    else:
        return parts

def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    ptm_list: List[str],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    This is a similar evaluation criterion as used by DeepNovo.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    ptm_matches_1 = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    ptm_matches_2 = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            idx = max(i1, i2)
            aa_matches[idx] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            if aa_matches[idx]:
                ptm_matches_1[idx] = (peptide1[i1] in ptm_list)
                ptm_matches_2[idx] = (peptide2[i2] in ptm_list)
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, ptm_matches_1, ptm_matches_2, aa_matches.all()


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    ptm_list: List[str],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    # Find longest mass-matching prefix.
    aa_matches, ptm_matches_1, ptm_matches_2, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, ptm_list, cum_mass_threshold, ind_mass_threshold
    )
    
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match, ptm_matches_1, ptm_matches_2
    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            idx = max(i1, i2)
            aa_matches[idx] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            if aa_matches[idx]:
                ptm_matches_1[idx] = (peptide1[i1] in ptm_list)
                ptm_matches_2[idx] = (peptide2[i2] in ptm_list)
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2

            
    return aa_matches, aa_matches.all(), ptm_matches_1, ptm_matches_2


def aa_match_batch(
    peptides1: Iterable,
    peptides2: Iterable,
    aa_dict: Dict[str, float],
    ptm_list = ['M(ox)','N(+.98)','Q(+.98)'],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : Iterable
        The first list of peptide sequences to be compared.
    peptides2 : Iterable
        The second list of peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    ptm_list : List[str]
        All the post-translational modification considered in validation.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match, 
        (iii) boolean flags indicating whether each ground truth amino acid with 
        ptm matches across both peptide sequences, (iv) boolean flags indicating 
        whether each predicted amino acid with ptm matches across both peptide 
        sequences,
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    n_ptm_1: int
        Total number of amino acids with post-translational modification in ground truth peptide sequences.
    n_ptm_2: int
        Total number of amino acids with post-translational modification in predicted peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    n_ptm_1, n_ptm_2 = 0, 0
    
    print("AA dict: ", aa_dict)
    print("PTM dict: ", ptm_list)
    for ptm in ptm_list:
        if ptm not in aa_dict.keys():
            raise ValueError(f"PTM type {ptm} is not in aa_dict: {aa_dict.keys()}!")
    
    for peptide1, peptide2 in zip(peptides1, peptides2):
        # Split peptides into individual AAs if necessary.
        if isinstance(peptide1, str):
            peptide1 = split_peptide(peptide1, aa_dict)
        if isinstance(peptide2, str):
            peptide2 = split_peptide(peptide2, aa_dict)
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)

        ptm_mask_1 = [1 if aa in ptm_list else 0 for aa in peptide1]
        ptm_mask_2 = [1 if aa in ptm_list else 0 for aa in peptide2]
        n_ptm_1 += sum(ptm_mask_1)
        n_ptm_2 += sum(ptm_mask_2)
        
        if len(peptide2) == 0:
            aa_matches_batch.append( (np.zeros(len(peptide1), np.bool_), False,
                                    np.zeros(len(peptide1), np.bool_),np.zeros(len(peptide1), np.bool_)) )
        else:
            aa_matches_batch.append(    # List[aa_matches, pep_matches, ptm_matches_1, ptm_matches_2]
                aa_match(
                    peptide1,
                    peptide2,
                    aa_dict,
                    ptm_list,
                    cum_mass_threshold,
                    ind_mass_threshold,
                    mode,
                )
            )
    return aa_matches_batch, n_aa1, n_aa2, n_ptm_1, n_ptm_2


def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
    n_ptm_true: int,
    n_ptm_pred: int,
    scores: List[float],
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match, 
        (iii) boolean flags indicating whether each ground truth amino acid with 
        ptm matches across both peptide sequences, (iv) boolean flags indicating 
        whether each predicted amino acid with ptm matches across both peptide 
        sequences,
    n_aa_true: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the second list of peptide sequences.
    n_ptm_true: int
        Total number of amino acids with post-translational modification in ground 
        truth peptide sequences.
    n_ptm_pred: int
        Total number of amino acids with post-translational modification in predicted 
        peptide sequences.
    scores: List [float]
        Confidence scores for every peptide predictions.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_precision: float
        The number of correct peptide predictions divided by the number of
        peptides.
    ptm_recall: float
        The number of correct AA with ptm predictions divided by the number of true AAs.
    ptm_precision: float
        The number of correct AA with ptm predictions divided by the number of predicted AAs.
    curve_auc: float
        Calculate area under curve of precision-recall(AUC).
    """
    print(f'PTM number: {n_ptm_true}  ;  {n_ptm_pred}')
    
    # aa_matches_batch: List[aa_matches, pep_matches, ptm_matches_1, ptm_matches_2]
    n_aa_correct = sum(
        [aa_matches[0].sum() for aa_matches in aa_matches_batch]
    )
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_precision = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )

    ptm_recall = sum([aa_matches[2].sum() for aa_matches in aa_matches_batch]) / (n_ptm_true + 1e-8)
    ptm_precision = sum([aa_matches[3].sum() for aa_matches in aa_matches_batch]) / (n_ptm_pred + 1e-8)

    # calculate precision-length relations 
    # pep_match_bool_list = [aa_matches[1] for aa_matches in aa_matches_batch]
    # prec_by_len = prec_by_length(pep_match_bool_list, length_list)  # Dict[int, float]
    
    # Calculate area under curve of precision-recall(AUC).
    pep_match_bool_list = [aa_matches[1] for aa_matches in aa_matches_batch]
    assert len(scores)==len(pep_match_bool_list)
    combined = list(zip(scores, pep_match_bool_list))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    sorted_pred_bools = [pred for _, pred in sorted_combined]
    
    precision = np.cumsum(sorted_pred_bools) / np.arange(1, len(sorted_pred_bools) + 1)
    recall = np.cumsum(sorted_pred_bools) / len(sorted_pred_bools)
    curve_auc = auc(recall, precision)
    
    metrics_dict = {
        "aa_precision" : aa_precision,
        "aa_recall" : aa_recall,
        "pep_precision" : pep_precision,
        "ptm_recall" : ptm_recall,
        "ptm_precision" : ptm_precision,
        "curve_auc" : curve_auc,
    }
    return metrics_dict


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total


if __name__ == '__main__':
    csv_path = sys.argv[1]

    df = pd.read_csv(csv_path, na_values=['', ' '])
    df = df.fillna('')
    match_ret = aa_match_batch(
            df['peptides_true'],
            df['peptides_pred'],
            STD_AA_MASS,
            # ['M(ox)','N(+.98)','Q(+.98)'],
            ['M(+15.99)','N(+.98)','Q(+.98)'],
            # ['M(ox)']
    )#  aa_matches_batch, n_aa1, n_aa2, n_ptm_1, n_ptm_2

    df['peptides_score'] = df['peptides_score'].replace(['nan', ''], np.nan)
    df['peptides_score'] = df['peptides_score'].astype(float).fillna(-np.inf)
    
    metrics_dict = aa_match_metrics(
        *match_ret, df['peptides_score']
    )#  metrics_dict = {"aa_precision" : float,"aa_recall" : float,"pep_precision" : float,"ptm_recall" : float,"ptm_precision" : float,"curve_auc" : float}

    df['pep_matches'] = [aa_matches[1] for aa_matches in match_ret[0]]
    df.to_csv(csv_path, index=False)
    
    print("—————————— Validation Results ——————————")
    for key, value in metrics_dict.items():
        print(f"{key}:\t {value}")
        

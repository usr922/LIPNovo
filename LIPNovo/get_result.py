import pandas as pd
from novobench.metrics.evaluate  import aa_match_batch,aa_match_metrics
import numpy as np



def get_result(filename):
    data = pd.read_csv(filename, header=None,na_values=['', ' '])
    data = data.fillna('')
    data.columns = ['peptides_true', 'peptides_pred', 'peptides_score']
    df = data

    mass_Phosphorylation = 79.96633
    STD_AA_MASS = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C(+57.02)": 160.030649,
    # "C": 160.030649 # V1
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
    "M(+15.99)": 147.035400,
    "N(+.98)": 115.026943,
    "Q(+.98)": 129.042594
    }

    match_ret = aa_match_batch(
            df['peptides_true'],
            df['peptides_pred'],
            STD_AA_MASS,
            ['M(+15.99)','N(+.98)','Q(+.98)'],
        # ['M+15.995','N+0.984','Q+0.984']
    )
    df['peptides_score'] = df['peptides_score'].replace(['nan', ''], np.nan)
    df['peptides_score'] = df['peptides_score'].astype(float).fillna(-np.inf)

    metrics_dict = aa_match_metrics(
        *match_ret, df['peptides_score']
    )#  metrics_dict = {"aa_precision" : float,"aa_recall" : float,"pep_precision" : float,"ptm_recall" : float,"ptm_precision" : float,"curve_auc" : float}


    print("—————————— Validation Results ——————————")
    for key, value in metrics_dict.items():
        print(f"{key}:\t {value}")




if __name__ == "__main__":

    get_result("result_csv")

from novobench.transforms.base import BaseTransform
from novobench.data.base import SpectrumData
import polars as pl


class AA_MAP_AA(BaseTransform):
    def __init__(self):
        super().__init__()
        self.mapper = {
            'C(+57.02)': 'C+57.021',
            'M(+15.99)': 'M+15.995',
            'N(+.98)': 'N+0.984',
            'Q(+.98)': 'Q+0.984'
        }

    def __call__(self, data: SpectrumData) -> SpectrumData:
        # Assuming data is a DataFrame and modified_sequence is a column in it
        # Iterate through the DataFrame and replace substrings in modified_sequence

        sequence = data.modified_sequence.apply(self.replace_sequence)
        data.set_df(data.df.with_columns([pl.Series("modified_sequence", sequence)]))
        return data

    def replace_sequence(self, sequence: str) -> str:
        # Replace each key in mapper with its corresponding value in the sequence
        for original, replacement in self.mapper.items():
            sequence = sequence.replace(original, replacement)
        return sequence
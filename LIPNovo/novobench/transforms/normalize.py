import spectrum_utils.spectrum as sus
from novobench.transforms.base import BaseTransform
from novobench.data.base import SpectrumData
import numpy as np
import polars as pl
import multiprocessing as mp
from tqdm import tqdm

def process_normalize(args):
    precursor_mz, precursor_charge, mz_array, int_array = args
    spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
    spectrum.scale_intensity("root", 1)
    intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
    mz = spectrum.mz
    intensity = intensities
    # 出现了空的情况, 检查一下m/z值是否正常
    if len(spectrum.mz) == 0:
        print('value error')
        mz = [0,]
        intensity = [1,]
    else:
        mz = mz.tolist()
        intensity = intensity.tolist()
    return mz, intensity
    

class ScaleIntensity(BaseTransform):
    """
    A transformation class that adjusts the m/z range of spectral data.
    
    Attributes:
        min_mz (float): The minimum m/z value to include in the spectrum.
        max_mz (float): The maximum m/z value to include in the spectrum.
    """

    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []


        with mp.Pool(processes=4) as pool:
            results = list(tqdm(pool.map(process_normalize, zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array))))
            
        for mz_lists, intensity_lists in results:
            updated_mz_arrays.append(mz_lists)
            updated_intensity_arrays.append(intensity_lists)
    

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays), 
                              pl.Series("intensity_array", updated_intensity_arrays)]))
        
        print('NORMALIZE DONE')
        return data
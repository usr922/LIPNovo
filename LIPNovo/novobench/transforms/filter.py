import spectrum_utils.spectrum as sus
from novobench.transforms.base import BaseTransform
from novobench.data.base import SpectrumData
import numpy as np
import polars as pl
import multiprocessing as mp
from tqdm import tqdm
def process_spectrum(args):
    precursor_mz, precursor_charge, mz_array, int_array,min_mz, max_mz = args
    spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
    spectrum.set_mz_range(min_mz, max_mz)
    mz = spectrum.mz
    intensity = spectrum.intensity
    if len(spectrum.mz) == 0:
        print('value error')
        mz = [0,]
        intensity = [1,]
    else:
        mz = mz.tolist()
        intensity = intensity.tolist()
    return mz, intensity


def process_precursor_peak(args):
    precursor_mz, precursor_charge, mz_array, int_array,remove_precursor_tol = args
    spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
    spectrum.remove_precursor_peak(remove_precursor_tol, "Da")
    mz = spectrum.mz
    intensity = spectrum.intensity
    if len(spectrum.mz) == 0:
        print('value error')
        mz = [0,]
        intensity = [1,]
    else:
        mz = mz.tolist()
        intensity = intensity.tolist()
    return mz, intensity

    
def process_filter_int(args):
    precursor_mz, precursor_charge, mz_array, int_array,min_intensity, n_peaks = args
    spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
    spectrum.filter_intensity(min_intensity, n_peaks)
    mz = spectrum.mz
    intensity = spectrum.intensity
    if len(spectrum.mz) == 0:
        print('value error')
        mz = [0,]
        intensity = [1,]
    else:
        mz = mz.tolist()
        intensity = intensity.tolist()
    return mz, intensity

class SetRangeMZ(BaseTransform):
    def __init__(self, min_mz: float = 50.0, max_mz: float = 2500.0):
        super().__init__()
        self.min_mz = min_mz
        self.max_mz = max_mz



    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []
        min_mz = [self.min_mz for i in range(len(data.precursor_mz))]
        max_mz = [self.max_mz for i in range(len(data.precursor_mz))]

        with mp.Pool(processes=4) as pool:
            results = list(tqdm(pool.map(process_spectrum, zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array,min_mz,max_mz))))
            
        for mz_lists, intensity_lists in results:
            updated_mz_arrays.append(mz_lists)
            updated_intensity_arrays.append(intensity_lists)
    

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays), 
                              pl.Series("intensity_array", updated_intensity_arrays)]))
        
        print('SET MZ RANGE DONE')
        return data
    
class RemovePrecursorPeak(BaseTransform):
    def __init__(self, remove_precursor_tol: float = 2.0):
        super().__init__()
        self.remove_precursor_tol = remove_precursor_tol


    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []
        remove_precursor_tol = [self.remove_precursor_tol for i in range(len(data.precursor_mz))]

        with mp.Pool(processes=4) as pool:
            results = list(tqdm(pool.map(process_precursor_peak, zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array,remove_precursor_tol))))
            
        for mz_lists, intensity_lists in results:
            updated_mz_arrays.append(mz_lists)
            updated_intensity_arrays.append(intensity_lists)
    

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays), 
                              pl.Series("intensity_array", updated_intensity_arrays)]))
        
        
        print('REMOVE PRECURSOR PEAK DONE')
        return data



class FilterIntensity(BaseTransform):
    def __init__(self, min_intensity: float = 0.01, n_peaks: int = 200):
        super().__init__()
        self.n_peaks = n_peaks
        self.min_intensity = min_intensity


    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []
        n_peaks = [self.n_peaks for i in range(len(data.precursor_mz))]
        min_intensity = [self.min_intensity for i in range(len(data.precursor_mz))]
        

        with mp.Pool(processes=4) as pool:
            results = list(tqdm(pool.map(process_filter_int, zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array,min_intensity,n_peaks))))
            
        for mz_lists, intensity_lists in results:
            updated_mz_arrays.append(mz_lists)
            updated_intensity_arrays.append(intensity_lists)
    

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays), 
                              pl.Series("intensity_array", updated_intensity_arrays)]))
        
        
        print('FILTER INTENSITY DONE')

        return data
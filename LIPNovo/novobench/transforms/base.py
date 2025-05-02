from abc import ABC, abstractmethod
from novobench.data.base import SpectrumData
import polars as pl

class BaseTransform(ABC):
    """BaseTransform abstract base class.

    This class defines the structure for transformations that can be applied to SpectrumData.
    """

    @property
    def name(self) -> str:
        """Returns the class name of the transform."""
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, data: SpectrumData) -> SpectrumData:
        raise NotImplementedError
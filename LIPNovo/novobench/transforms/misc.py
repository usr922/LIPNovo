from typing import Tuple
from novobench.transforms.base import BaseTransform
import polars as pl

class Compose(BaseTransform):
    """Compose transformation by combining several transformation objects.

    This class allows for the sequential application of multiple transformation objects to a data object.

    Parameters
    ----------
    transforms : Tuple[BaseTransform, ...]
        A tuple containing transformation objects that inherit from BaseTransform.

    Raises
    ------
    TypeError
        If any of the provided transformations do not inherit from BaseTransform.

    Notes
    -----
    The order in which the `transforms` are provided is the order in which they will be applied to the data object.

    Examples
    --------
    To use Compose, instantiate it with transformation objects:

        >>> composed_transform = Compose(Transform1(), Transform2(), Transform3())
        >>> transformed_data = composed_transform(data_object)

    Where `Transform1`, `Transform2`, `Transform3` are all classes that inherit from `BaseTransform`.
    """

    def __init__(self, *transforms: Tuple[BaseTransform, ...], **kwargs):
        super().__init__(**kwargs)

        failed_list = [transform for transform in transforms if not isinstance(transform, BaseTransform)]
        if failed_list:
            failed_types_str = ", ".join(f"{type(transform).__name__}" for transform in failed_list)
            raise TypeError(f"All transform objects must inherit from BaseTransform. The following have incorrect types: {failed_types_str}")

        self.transforms = transforms

    def __repr__(self) -> str:
        transform_repr_str = ", ".join(repr(transform) for transform in self.transforms)
        return f"Compose({transform_repr_str})"

    def __getitem__(self, idx: int) -> BaseTransform:
        return self.transforms[idx]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def __call__(self, data) -> None:
        for transform in self.transforms:
            data = transform(data)
        return data

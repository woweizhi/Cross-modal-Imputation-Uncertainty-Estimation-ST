from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import anndata
import scanpy as sc

from torch.utils.data import Sampler, DataLoader
import numpy as np

from anndata.experimental import AnnLoader

from copy import copy
from functools import partial
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse

from anndata import AnnData
from anndata.experimental.multi_files._anncollection import AnnCollection, _ConcatViewMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import torch
    from torch.utils.data import BatchSampler, DataLoader, Sampler
except ImportError:
    Sampler, BatchSampler, DataLoader = object, object, object


# Custom sampler to get proper batches instead of joined separate indices
# maybe move to multi_files
class BatchIndexSampler(Sampler):
    def __init__(self, n_obs, batch_size, shuffle=False, drop_last=False):
        self.n_obs = n_obs
        self.batch_size = batch_size if batch_size < n_obs else n_obs
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.n_obs).tolist()
        else:
            indices = list(range(self.n_obs))

        for i in range(0, self.n_obs, self.batch_size):
            batch = indices[i : min(i + self.batch_size, self.n_obs)]

            # only happens if the last batch is smaller than batch_size
            if len(batch) < self.batch_size and self.drop_last:
                continue

            yield batch

    def __len__(self):
        if self.drop_last:
            length = self.n_obs // self.batch_size
        else:
            length = ceil(self.n_obs / self.batch_size)

        return length



# maybe replace use_cuda with explicit device option
# TODO: adopted from anndata.experimental, change explicitly use_cuda to device
def default_converter(arr, device, pin_memory):
    if isinstance(arr, torch.Tensor):
        if device:
            arr = arr.to(device)
        elif pin_memory:
            arr = arr.pin_memory()
    elif arr.dtype.name != "category" and np.issubdtype(arr.dtype, np.number):
        if issparse(arr):
            arr = arr.toarray()
        if device:
            arr = torch.tensor(arr, device=device)
        else:
            arr = torch.tensor(arr)
            arr = arr.pin_memory() if pin_memory else arr
    return arr


def _convert_on_top(convert, top_convert, attrs_keys):
    if convert is None:
        new_convert = top_convert
    elif callable(convert):

        def compose_convert(arr):
            return top_convert(convert(arr))

        new_convert = compose_convert
    else:
        new_convert = {}
        for attr in attrs_keys:
            if attr not in convert:
                new_convert[attr] = top_convert
            else:
                if isinstance(attrs_keys, list):
                    as_ks = None
                else:
                    as_ks = attrs_keys[attr]
                new_convert[attr] = _convert_on_top(convert[attr], top_convert, as_ks)
    return new_convert


# AnnLoader has the same arguments as DataLoader, but uses BatchIndexSampler by default
class AnnLoader(DataLoader):
    """\
    PyTorch DataLoader for AnnData objects.

    Builds DataLoader from a sequence of AnnData objects, from an
    :class:`~anndata.experimental.AnnCollection` object or from an `AnnCollectionView` object.
    Takes care of the required conversions.

    Parameters
    ----------
    adatas
        `AnnData` objects or an `AnnCollection` object from which to load the data.
    batch_size
        How many samples per batch to load.
    shuffle
        Set to `True` to have the data reshuffled at every epoch.
    use_default_converter
        Use the default converter to convert arrays to pytorch tensors, transfer to
        the default cuda device (if `use_cuda=True`), do memory pinning (if `pin_memory=True`).
        If you pass an AnnCollection object with prespecified converters, the default converter
        won't overwrite these converters but will be applied on top of them.
    use_cuda
        Transfer pytorch tensors to the default cuda device after conversion.
        Only works if `use_default_converter=True`
    **kwargs
        Arguments for PyTorch DataLoader. If `adatas` is not an `AnnCollection` object, then also
        arguments for `AnnCollection` initialization.
    """

    def __init__(
        self,
        adatas: Sequence[AnnData] | dict[str, AnnData],
        batch_size: int = 1,
        shuffle: bool = False,
        use_default_converter: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        if isinstance(adatas, AnnData):
            adatas = [adatas]

        if (
            isinstance(adatas, list)
            or isinstance(adatas, tuple)
            or isinstance(adatas, dict)
        ):
            join_obs = kwargs.pop("join_obs", "inner")
            join_obsm = kwargs.pop("join_obsm", None)
            label = kwargs.pop("label", None)
            keys = kwargs.pop("keys", None)
            index_unique = kwargs.pop("index_unique", None)
            convert = kwargs.pop("convert", None)
            harmonize_dtypes = kwargs.pop("harmonize_dtypes", True)
            indices_strict = kwargs.pop("indices_strict", True)

            dataset = AnnCollection(
                adatas,
                join_obs=join_obs,
                join_obsm=join_obsm,
                label=label,
                keys=keys,
                index_unique=index_unique,
                convert=convert,
                harmonize_dtypes=harmonize_dtypes,
                indices_strict=indices_strict,
            )

        elif isinstance(adatas, _ConcatViewMixin):
            dataset = copy(adatas)
        else:
            raise ValueError("adata should be of type AnnData or AnnCollection.")

        if use_default_converter:
            pin_memory = kwargs.pop("pin_memory", False)
            _converter = partial(
                default_converter, device=device, pin_memory=pin_memory
            )
            dataset.convert = _convert_on_top(
                dataset.convert, _converter, dict(dataset.attrs_keys, X=[])
            )

        has_sampler = "sampler" in kwargs
        has_batch_sampler = "batch_sampler" in kwargs

        has_worker_init_fn = (
            "worker_init_fn" in kwargs and kwargs["worker_init_fn"] is not None
        )
        has_workers = "num_workers" in kwargs and kwargs["num_workers"] > 0
        use_parallel = has_worker_init_fn or has_workers

        if (
            batch_size is not None
            and batch_size > 1
            and not has_batch_sampler
            and not use_parallel
        ):
            drop_last = kwargs.pop("drop_last", False)

            if has_sampler:
                sampler = kwargs.pop("sampler")
                sampler = BatchSampler(
                    sampler, batch_size=batch_size, drop_last=drop_last
                )
            else:
                sampler = BatchIndexSampler(
                    len(dataset), batch_size, shuffle, drop_last
                )

            super().__init__(dataset, batch_size=None, sampler=sampler, **kwargs)
        else:
            super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        """
        Args:
            data_source (Dataset): Dataset to sample from.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the data before sampling.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data_source)

    def __iter__(self):
        """
        Returns an iterator that yields batches of indices.
        """
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)  # Shuffle indices if required

        # Create batches by splitting the indices
        batches = [indices[i:i + self.batch_size] for i in range(0, self.num_samples, self.batch_size)]

        return iter(batches)

    def __len__(self):
        """
        Returns the total number of batches.
        """
        return (self.num_samples + self.batch_size - 1) // self.batch_size  # Ceiling division to account for leftover samples

class AnnDataset(Dataset):
    def __init__(self, adata):
        """
        Args:
            adata (AnnData): AnnData object containing the data
        """
        self.adata = adata

    def __len__(self):
        """
        Returns the total number of samples
        """
        return self.adata.shape[0]

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset at index `idx`
        """
        # Extract the data from the .X matrix (gene expression)
        data = self.adata[idx]

        return data

        # # You can also extract other relevant information like labels from .obs if available
        # #TODO: would be better to adopt AnnDataView object as AnnaDataLoader did:
        # label = self.adata.obs.iloc[idx].get('label', None)  # Replace 'label' with your column name
        #
        # # If the data is sparse, convert it to a dense format
        # if hasattr(data, "toarray"):
        #     data = data.toarray()
        #
        # # Convert the data to a PyTorch tensor
        # data = torch.tensor(data, dtype=torch.float32)
        #
        # if label is not None:
        #     label = torch.tensor(label, dtype=torch.long)
        #     return data, label
        # else:
        #     return data

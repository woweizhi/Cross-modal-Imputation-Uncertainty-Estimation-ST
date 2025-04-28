import logging
import os
import sys
import warnings
from typing import List, Optional, Union, Tuple
from math import ceil
import random
import numpy as np
import torch
from anndata import AnnData
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Sampler
from data.anna_dataloader import AnnLoader

from scvi import REGISTRY_KEYS, settings
from scvi.data.fields import CategoricalObsField, LayerField, NumericalObsField
from scvi.data import AnnDataManager
from scvi.model._utils import parse_device_args
from scvi.model.base import BaseModelClass, VAEMixin
from scvi.train import Trainer
from scvi.utils._docstrings import devices_dsp
from data import data_processing
from lightning.pytorch.utilities import CombinedLoader
from scipy.special import softmax

sys.path.append("../")

from module._module import DCVAE
from src._task import CTLTrainingPlan

logger = logging.getLogger(__name__)

class BatchIndexSampler(Sampler):
    def __init__(self, n_obs, batch_size, shuffle=False, drop_last=False):
        self.n_obs = n_obs
        self.batch_size = min(batch_size, n_obs)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = np.random.permutation(self.n_obs) if shuffle else np.arange(self.n_obs)

    def __iter__(self):
        for i in range(0, self.n_obs, self.batch_size):
            batch = self.indices[i:i+self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch.tolist()

    def __len__(self):
        return self.n_obs // self.batch_size if self.drop_last else ceil(self.n_obs / self.batch_size)


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

class ContrastSampler(Sampler):
    def __init__(self, sampler, batch_size, topK_ind_matrix, topK_contrast):
        """
        Args:
            sampler (Sampler): Sampler providing indices from another modality.
            batch_size (int): Batch size for current modality (should match sampler batch size).
            topK_ind_matrix (np.ndarray): (num_items x topK_contrast) array of indices for positive pairs.
            topK_contrast (int): Number of top correlated items to choose from.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.topK_ind_matrix = topK_ind_matrix
        self.topK_contrast = topK_contrast

        # Correct way to calculate length (should match sampler length)
        self._length = len(sampler)

    def __iter__(self):
        for ind in self.sampler:
            # Randomly select positive pair indices from topK correlated indices
            generate_pos_pair = np.random.randint(0, self.topK_contrast, size=len(ind))
            batch_ind = self.topK_ind_matrix[ind, generate_pos_pair]

            yield batch_ind.tolist()

    def __len__(self):
        return self._length

class CrossModalSampler(Sampler):
    def __init__(self, sampler, batch_size, corr_matrix, axis=1):
        """
        Args:
            sampler (Sampler): Sampler from the other modality.
            batch_size (int): Batch size for current modality.
            corr_matrix (np.ndarray): Pearson correlation matrix between modalities.
            axis (int): Axis along which to take indices from corr_matrix.
                        axis=1 means the rows correspond to the current modality,
                        axis=0 means columns correspond to the current modality.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.corr_matrix = corr_matrix
        self.axis = axis

        # Pre-calculate length (static, not dynamic)
        self._length = len(sampler)

    def __iter__(self):
        for ind in self.sampler:
            # Get the sub-matrix according to indices from other modality
            sub_corr = np.take(self.corr_matrix, ind, axis=self.axis)

            # Compute mean correlation across selected indices
            sub_corr_mean = np.mean(sub_corr, axis=self.axis)

            # Softmax normalization to get sampling probabilities
            sub_corr_prob = softmax(sub_corr_mean)

            all_indices = np.arange(len(sub_corr_mean))

            # Sample indices according to probabilities (with replacement)
            batch_ind = np.random.choice(
                all_indices, len(ind), p=sub_corr_prob, replace=True
            ).tolist()

            yield batch_ind

    def __len__(self):
        # length determined by the number of batches from the other modality
        return self._length

def _unpack_tensors(tensors):
    x = tensors[REGISTRY_KEYS.X_KEY].squeeze_(0)
    batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze_(0)
    y = tensors[REGISTRY_KEYS.LABELS_KEY].squeeze_(0)
    return x, batch_index, y

def _init_library_size(
    data, n_batch: dict
) -> Tuple[np.ndarray, np.ndarray]:

    batch_indices = data.obs["batch"]

    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i_batch in np.unique(batch_indices):
        idx_batch = np.squeeze(batch_indices == i_batch)
        batch_data = data[
            idx_batch.to_numpy().nonzero()[0]
        ]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.X.sum(axis=1)
        # Operations on numpy masked array gives invalid values masked
        # masked_array(data=[-- -- 0.0 0.69314718056],
        #              mask=[True  True False False],
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            warnings.warn(
                "This dataset has some empty cells, this might fail inference."
                "Data should be filtered with `scanpy.pp.filter_cells()`",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        # Return input as an array with masked data replaced by a fill value.
        log_counts = masked_log_sum.filled(0)
        library_log_means[i_batch] = np.mean(log_counts).astype(np.float32)
        library_log_vars[i_batch] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(1, -1), library_log_vars.reshape(1, -1)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class STFormer(VAEMixin, BaseModelClass):

    def __init__(
        self,
        adata_seq: AnnData,
        adata_spatial: AnnData,
        generative_distributions: Optional[List[str]] = None,
        model_library_size: Optional[List[bool]] = None,
        n_latent: int = 20,
        corr_metric: str = "Pearson",
        topK_contrastive: int = 50,
        CL_weight: float = 5.0,
        device: str = "cuda",
        **model_kwargs,
    ):
        super().__init__()
        if adata_seq is adata_spatial:
            raise ValueError(
                "`adata_seq` and `adata_spatial` cannot point to the same object. "
                "If you would really like to do this, make a copy of the object and pass it in as `adata_spatial`."
            )

        seed_torch()

        model_library_size = model_library_size or [False, False]
        generative_distributions = generative_distributions or ["zinb", "nb"]
        self.adatas = [adata_seq, adata_spatial]

        self.registries_ = []

        seq_var_names = adata_seq.var_names
        spatial_var_names = adata_spatial.var_names

        if not set(spatial_var_names) <= set(seq_var_names):
            raise ValueError("spatial genes needs to be subset of seq genes")

        spatial_gene_loc = [
            np.argwhere(seq_var_names == g)[0] for g in spatial_var_names
        ]
        spatial_gene_loc = np.concatenate(spatial_gene_loc)
        gene_mappings = [slice(None), spatial_gene_loc]

        n_inputs = [len(seq_var_names), len(spatial_var_names)]
        total_genes = n_inputs[0]

        n_batches = 2
        #
        library_log_means = []
        library_log_vars = []
        for adata in self.adatas:
            adata_library_log_means, adata_library_log_vars = _init_library_size(
                adata, n_batches
            )
            library_log_means.append(adata_library_log_means)
            library_log_vars.append(adata_library_log_vars)

        ############ calculate the correlation matrix between scRNA-seq and ST ###########
        self.corr_matrix_raw = data_processing.calculate_corr(adata_seq, adata_spatial, spatial_var_names, corr_metric)

        n_samples = [adata_seq.shape[0], adata_spatial.shape[0]]
        ind = np.argmin(n_samples)

        #self.device = device

        scale_topK_contrastive = topK_contrastive * np.array(n_samples)/n_samples[ind]
        self.topK_contra_sc, self.topK_contra_st = scale_topK_contrastive.astype(int)

        _, self.topK_ind_sc_contra = data_processing.get_sample_weights(corr_matrix=self.corr_matrix_raw, topK=int(scale_topK_contrastive[0]), axis=0)
        _, self.topK_ind_st_contra = data_processing.get_sample_weights(corr_matrix=self.corr_matrix_raw, topK=int(scale_topK_contrastive[1]), axis=1)

        # initialize the binary mask among topK cells for every given spot and topK spots for every given cells
        topK_binary_matrix = torch.zeros(adata_seq.shape[0], adata_spatial.shape[0])
        topK_binary_matrix_sc = topK_binary_matrix.scatter(0, torch.tensor(self.topK_ind_sc_contra), 1)
        topK_binary_matrix_st = topK_binary_matrix.scatter(1, torch.tensor(self.topK_ind_st_contra), 1)

        self.corr_matrix = torch.from_numpy(self.corr_matrix_raw).float().to(device)
        self.topK_binary_matrix_sc = torch.gt(topK_binary_matrix_sc, 0).to(device)
        self.topK_binary_matrix_st = torch.gt(topK_binary_matrix_st, 0).to(device)

        self.module = DCVAE(
            n_inputs,
            total_genes,
            gene_mappings,
            generative_distributions,
            model_library_size,
            library_log_means,
            library_log_vars,
            self.corr_matrix,
            self.topK_binary_matrix_sc,
            self.topK_binary_matrix_st,
            n_batch=n_batches,
            n_latent=n_latent,
            topK_contrastive=topK_contrastive,
            CL_weight=CL_weight,
            **model_kwargs,
        )

        #self.device = device
        self.to_device(device)
        self.init_params_ = self._get_init_params(locals())

    #@devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 200,
        use_gpu: Optional[Union[str, int, bool]] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        shuffle_set_split: bool = True,
        batch_size: int = 1024,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):

        accelerator, devices, device = parse_device_args(
            use_gpu=use_gpu,
            accelerator=accelerator,
            devices=devices,
            return_device="torch",
        )

        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **kwargs,
        )

        ######preparing the dataloader for torch model
        adata_seq, adata_spatial = self.adatas

        use_cuda = torch.cuda.is_available()

        # TODO: for random sample cells, spots in cross attention and spots in contrastive learning are all conditional on randomly selected cells
        rand_sampler_sc = BatchIndexSampler(adata_seq.shape[0], batch_size=batch_size, shuffle=True, drop_last=True)
        cross_sampler_st = CrossModalSampler(rand_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        contra_sampler_st = ContrastSampler(rand_sampler_sc, batch_size, self.topK_ind_st_contra, self.topK_contra_st)
        contra_cross_sampler_sc = CrossModalSampler(contra_sampler_st, batch_size, self.corr_matrix_raw, 1)

        # TODO: for random sample spots, cells in cross attention and cells in contrastive learning are all conditional on randomly selected spots
        rand_sampler_st = BatchIndexSampler(adata_spatial.shape[0], batch_size=batch_size, shuffle=True, drop_last=True)
        cross_sampler_sc = CrossModalSampler(rand_sampler_st, batch_size, self.corr_matrix_raw, 1)

        contra_sampler_sc = ContrastSampler(rand_sampler_st, batch_size, self.topK_ind_sc_contra.T, self.topK_contra_sc)
        contra_cross_sampler_st = CrossModalSampler(contra_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        # construct dataloader for randomly selected cells
        dl_rand_sc = AnnLoader(adata_seq, sampler=rand_sampler_sc, batch_size=None, device=self.device)
        dl_cross_st = AnnLoader(adata_spatial, sampler=cross_sampler_st, batch_size=None, device=self.device)

        dl_contra_st = AnnLoader(adata_spatial, sampler=contra_sampler_st, batch_size=None, device=self.device)
        dl_contra_cross_sc = AnnLoader(adata_seq, sampler=contra_cross_sampler_sc, batch_size=None, device=self.device)

        # construct dataloader for randomly selected spots
        dl_rand_st = AnnLoader(adata_spatial, sampler=rand_sampler_st, batch_size=None, device=self.device)
        dl_cross_sc = AnnLoader(adata_seq, sampler=cross_sampler_sc, batch_size=None,device=self.device)

        dl_contra_sc = AnnLoader(adata_seq, sampler=contra_sampler_sc, batch_size=None,device=self.device)
        dl_contra_cross_st = AnnLoader(adata_spatial, sampler=contra_cross_sampler_st, batch_size=None, device=self.device)

        rand_sc_dl = [[dl_rand_sc, dl_cross_st], [dl_contra_cross_sc, dl_contra_st]]
        rand_st_dl = [[dl_contra_sc, dl_contra_cross_st], [dl_cross_sc, dl_rand_st]]


        # two dataloaders from two different datasets within-the-same-loop
        # the shorter one will keep iterating the dataset until reach the max batches of the bigger one
        # https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/5
        train_dl = {"sc": rand_sc_dl, "st": rand_st_dl}
        train_dl = CombinedLoader(train_dl, mode="max_size_cycle")

        ## change adversarial classifier to False
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        self._training_plan = CTLTrainingPlan(
            self.module,
            **plan_kwargs,
        )

        self.trainer.fit(self._training_plan, train_dl)

        try:
            self.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None
        self.module.eval()

        self.to_device(device)
        self.is_trained_ = True

    def _make_scvi_dls(self, adatas: List[AnnData] = None, sample_weights: List[np.ndarray]=None, batch_size=128):
        if adatas is None:
            adatas = self.adatas
        post_list = [self._make_data_loader(ad, sampler=WeightedRandomSampler(sample_weights[i], len(sample_weights[i]))) for i,ad in enumerate(adatas)]
        for i, dl in enumerate(post_list):
            dl.mode = i

        return post_list

    @torch.inference_mode()
    def get_px_para(self, batch_size, symbol):
        # symbol is one of the values in the following:
        # px_scale: normalized gene expression frequency
        # px_rate: px_scale * exp(library)
        # px_r: dispersion parameter
        # px_dropout: dropout rate

        self.module.eval()

        adata_seq, adata_spatial = self.adatas

        ############### load data from corresponding index for every batch in the other modality
        rand_sampler_st = BatchIndexSampler(adata_spatial.shape[0], batch_size=batch_size, shuffle=False, drop_last=False)
        rand_sampler_sc = BatchIndexSampler(adata_seq.shape[0], batch_size=batch_size, shuffle=False, drop_last=False)

        cross_sampler_sc = CrossModalSampler(rand_sampler_st, batch_size, self.corr_matrix_raw, 1)
        cross_sampler_st = CrossModalSampler(rand_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        dataloader_st = AnnLoader(adata_spatial, sampler=rand_sampler_st, batch_size=None, device=self.device)
        dataloader_sc = AnnLoader(adata_seq, sampler=rand_sampler_sc, batch_size=None, device=self.device)

        dataloader_cross_st = AnnLoader(adata_spatial, sampler=cross_sampler_st, batch_size=None, device=self.device)
        dataloader_cross_sc = AnnLoader(adata_seq, sampler=cross_sampler_sc, batch_size=None, device=self.device)

        sc_dl = data_processing.TrainDL([dataloader_sc, dataloader_cross_st])
        st_dl = data_processing.TrainDL([dataloader_cross_sc, dataloader_st])

        val_dl = {"sc": sc_dl, "st": st_dl}

        retrive_values = []
        for mode, key in enumerate(["sc", "st"]):
            retrive_value = []
            dl = val_dl[key]
            for i,  batch in enumerate(dl): #
                scdl1, scdl2 = batch
                # **************** feed non-negative weights ********************#
                ind_row = scdl1.obs["index"]  # .squeeze().int().to("cuda")#.to(torch.long)
                ind_col = scdl2.obs["index"]  # .squeeze().int().to("cuda")#.to(torch.long)

                M1 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
                M2 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)

                dls = [scdl1, scdl2]

                retrive_value.append(
                    self.module._run_forward(
                        [scdl1.X.float(), scdl2.X.float()],
                        mode,
                        dls[mode].obs["batch"],  # [mode],
                        dls[mode].obs["labels"],  # [mode],
                        deterministic=True,
                        decode_mode=None,
                    )[symbol]
                )

            retrive_value = torch.cat(retrive_value).cpu().detach().numpy()
            retrive_values.append(retrive_value)

        return (retrive_values)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 1024,
    ) -> List[np.ndarray]:
        """Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas

        rand_sampler_st = BatchIndexSampler(adatas[1].shape[0], batch_size=batch_size, shuffle=False, drop_last=False)
        rand_sampler_sc = BatchIndexSampler(adatas[0].shape[0], batch_size=batch_size, shuffle=False, drop_last=False)

        cross_sampler_sc = CrossModalSampler(rand_sampler_st, batch_size, self.corr_matrix_raw, 1)
        cross_sampler_st = CrossModalSampler(rand_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        dataloader_st = AnnLoader(adatas[1], sampler=rand_sampler_st, batch_size=None, device=self.device)
        dataloader_sc = AnnLoader(adatas[0], sampler=rand_sampler_sc, batch_size=None, device=self.device)

        dataloader_cross_st = AnnLoader(adatas[1], sampler=cross_sampler_st, batch_size=None, device=self.device)
        dataloader_cross_sc = AnnLoader(adatas[0], sampler=cross_sampler_sc, batch_size=None, device=self.device)

        sc_dl = data_processing.TrainDL([dataloader_sc, dataloader_cross_st])
        st_dl = data_processing.TrainDL([dataloader_cross_sc, dataloader_st])

        val_dl = {"sc": sc_dl, "st": st_dl}

        self.module.eval()
        latents = []

        for mode, key in enumerate(["sc", "st"]):
            latent = []
            dl = val_dl[key]
            for i,  batch in enumerate(dl): #
                scdl1, scdl2 = batch

                ind_row = scdl1.obs["index"]#.squeeze().int().to("cuda")#.to(torch.long)
                ind_col = scdl2.obs["index"]#.squeeze().int().to("cuda")#.to(torch.long)

                M1 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
                M2 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)

                latent.append(self.module.sample_from_posterior_z([scdl1.X.float(), scdl2.X.float()], mode, deterministic=deterministic))
            latent = torch.cat(latent).cpu().detach().numpy()
            latents.append(latent)

        return latents

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        index: Optional[str] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, index),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
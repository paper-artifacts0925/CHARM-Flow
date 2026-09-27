"""Module `data/dataset_io.py`."""
import h5py
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def row_from_obsm(dataset, h5f: h5py.File, ds_name: str, row_idx: int) -> torch.Tensor:
    """
    Row from obsm.

    :param dataset: Input `dataset` value.
    :param h5f: Input `h5f` value.
    :param ds_name: Dataset name identifier.
    :param row_idx: Index value used for lookup or slicing.
    :return: Computed output(s) for this function.
    """
    if "obsm" not in h5f:
        raise KeyError("no obsm in h5/h5ad file")
    obsm = h5f["obsm"]

    key = dataset.data_args.embed_key
    if key not in obsm:
        avail = [
            k.decode("utf-8", "ignore") if isinstance(k, (bytes, bytearray)) else str(k)
            for k in obsm.keys()
        ]
        raise KeyError(f"'obsm/{key}' not exists. Available obsm keys are: {avail}")

    obj = obsm[key]
    enc = obj.attrs.get("encoding-type", None)
    if isinstance(enc, (bytes, bytearray)):
        enc = enc.decode("utf-8", "ignore")

    # dense
    if enc == "array":
        row = torch.as_tensor(obj[row_idx], dtype=torch.float32)
        return row.unsqueeze(0)
    elif isinstance(obj, h5py.Group):
        if all(k in obj for k in ("data", "indices", "indptr")):
            indptr = obj["indptr"]
            start = int(indptr[row_idx])
            end = int(indptr[row_idx + 1])
            cols = obj["indices"][start:end]
            vals = obj["data"][start:end]

            if ds_name not in dataset._obsm_dim:
                shp = obj.attrs.get("shape", None)
                if shp is None and "shape" in obj:
                    shp = obj["shape"][()]
                if shp is None:
                    raise KeyError(f"Cannot infer 'obsm/{key}' 's column number")
                dataset._obsm_dim[ds_name] = int(shp[1])

            d = dataset._obsm_dim[ds_name]
            try:
                assert d == dataset.data_args.embed_shape
            except:
                logger.warning("obs dim: %s", d)

            crow = torch.tensor([0, end - start], dtype=torch.int64)
            ccol = torch.as_tensor(cols, dtype=torch.int64)
            cdat = torch.as_tensor(vals, dtype=torch.float32)
            dense = torch.sparse_csr_tensor(crow, ccol, cdat, size=(1, d)).to_dense()
            return dense

    raise KeyError(f"Cannot identify obsm/{key}, not an arry nor csr_matrix")


def row_from_X(dataset, h5f: h5py.File, ds_name: str, row_idx: int) -> torch.Tensor:
    """
    Row from x.

    :param dataset: Input `dataset` value.
    :param h5f: Input `h5f` value.
    :param ds_name: Dataset name identifier.
    :param row_idx: Index value used for lookup or slicing.
    :return: Computed output(s) for this function.
    """
    X = h5f["/X"] if "/X" in h5f else h5f["X"]
    enc = X.attrs.get("encoding-type", None)
    if isinstance(enc, (bytes, bytearray)):
        enc = enc.decode("utf-8", "ignore")

    if enc == "csr_matrix":
        indptrs = h5f["/X/indptr"]
        start_ptr = int(indptrs[row_idx])
        end_ptr = int(indptrs[row_idx + 1])
        length = end_ptr - start_ptr
        sub_data = torch.empty(length, dtype=torch.float32)
        sub_indices = torch.empty(length, dtype=torch.int32)
        h5f["/X/data"].read_direct(sub_data.numpy(), np.s_[start_ptr:end_ptr])
        h5f["/X/indices"].read_direct(sub_indices.numpy(), np.s_[start_ptr:end_ptr])

        n_genes = (
            int(X.attrs["shape"][1])
            if "shape" in X.attrs
            else int(h5f["/X/shape"][1])
            if "/X/shape" in h5f
            else int(h5f["X"].shape[1])
            if isinstance(X, h5py.Dataset)
            else None
        )
        if n_genes is None:
            raise KeyError(f"Cannot infer n_genes for dataset '{ds_name}'")

        counts = torch.sparse_csr_tensor(
            [0],
            sub_indices,
            sub_data,
            (1, n_genes),
        )
        counts = counts.to_dense()
    else:
        counts = torch.tensor(h5f["X"][row_idx]).unsqueeze(0)

    return counts


def retrieve_counts(dataset, h5f, ds_name, local_idx):
    """
    Retrieve counts.

    :param dataset: Input `dataset` value.
    :param h5f: Input `h5f` value.
    :param ds_name: Dataset name identifier.
    :param local_idx: Index value used for lookup or slicing.
    :return: Computed output(s) for this function.
    """
    try:
        if dataset.data_args.embed_key.startswith("X_") and dataset.selected_gene_mask[ds_name].sum() == 2000:
            counts = row_from_obsm(dataset, h5f, ds_name, local_idx)
            return counts
        elif dataset.data_args.embed_key == "X":
            counts = row_from_X(dataset, h5f, ds_name, local_idx)
        else:
            raise NotImplementedError
    except Exception as iex:
        dataset.py_logger.exception(f"Error in dataset {ds_name}")
        raise iex

    all_genes = dataset.gene_vars[ds_name]
    mask = dataset.selected_gene_mask[ds_name]
    counts = counts[:, mask]
    cur_genes = all_genes[mask].tolist()
    cur_index = {g: i for i, g in enumerate(cur_genes)}
    if isinstance(dataset.selected_genes, dict):
        column_map = np.array([cur_index.get(g, -1) for g in dataset.selected_genes[ds_name]])
        new_length = len(dataset.selected_genes[ds_name])
    else:
        column_map = np.array([cur_index.get(g, -1) for g in dataset.selected_genes])
        new_length = len(dataset.selected_genes)

    existing_mask = column_map >= 0
    real_counts = torch.zeros(counts.shape[0], new_length, dtype=counts.dtype)
    real_counts[:, existing_mask] = counts[:, column_map[existing_mask]]

    return real_counts

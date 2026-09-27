"""Split strategy helpers split from data_module.py (logic-preserving)."""

import numpy as np
from tqdm import tqdm

def split_pbmc(dm, cache, target_type, key_info, holdout_setname=None, random_setname=None):
    """
    Split pbmc.

    :param dm: Input `dm` value.
    :param cache: Input `cache` value.
    :param target_type: Input `target_type` value.
    :param key_info: Input `key_info` value.
    :param holdout_setname: Input `holdout_setname` value.
    :param random_setname: Input `random_setname` value.
    :return: Computed output(s) for this function.
    """
    assert holdout_setname == "holdout_pert" and random_setname == "random_Perturbseq"

    left_indices = np.arange(cache.n_cells)
    categories = cache.pert_categories
    codes = cache.pert_codes
    control_code = (key_info["control_pert"] == categories).nonzero()[0]
    assert len(control_code) == 1
    control_mask = codes == control_code
    control_indices = left_indices[control_mask]
    left_indices = left_indices[~control_mask]

    holdout_indices = {"validation": [], "test": []}

    if dm.data_args.split_control:
        splited_control_indices = {"train": [], "validation": [], "test": []}

    assert len(key_info["holdout_batches"]) != 0
    for tgt_batch_idx, tgt_batch in enumerate(tqdm(cache.batch_categories)):
        if tgt_batch not in key_info["holdout_batches"]:
            continue
        tgt_batch_mask = cache.batch_codes == tgt_batch_idx
        if not (tmp_n_cells := np.sum(tgt_batch_mask)):
            continue
        tgt_batch_indices = np.where(tgt_batch_mask)[0]

        n_cells = {"validation": 0, "test": 0}
        for split in dm.all_split_names:
            for tgt_pert in key_info["holdout_pert"][split]:
                tgt_pert_idx = {v: k for k, v in enumerate(cache.pert_categories)}[tgt_pert]

                tgt_pert_mask = cache.pert_codes == tgt_pert_idx
                if not (tmp_n_cells := np.sum(tgt_pert_mask)):
                    continue
                tgt_pert_indices = np.where(tgt_pert_mask)[0]

                shared_indices = tgt_batch_indices[np.isin(tgt_batch_indices, tgt_pert_indices)]
                holdout_indices[split].append(shared_indices)
                n_cells[split] += len(shared_indices)

        if dm.data_args.split_control:
            ctrl_mask_for_batch = cache.pert_codes[tgt_batch_indices] == control_code
            ctrl_indices_for_batch = tgt_batch_indices[ctrl_mask_for_batch]

            n_val, n_test = n_cells["validation"], n_cells["test"]
            N = len(tgt_batch_indices)
            n_train = N - n_val - n_test
            assert n_train > 0

            n_ctrl_val = int(len(ctrl_indices_for_batch) * n_val / N)
            n_ctrl_test = int(len(ctrl_indices_for_batch) * n_test / N)

            rng = np.random.default_rng(dm.seed)
            ctrl_indices_for_batch_shuffled = rng.permutation(ctrl_indices_for_batch)
            val_ctrl_indices = ctrl_indices_for_batch_shuffled[:n_ctrl_val]
            test_ctrl_indices = ctrl_indices_for_batch_shuffled[n_ctrl_val:n_ctrl_val + n_ctrl_test]

            splited_control_indices["validation"].append(val_ctrl_indices)
            splited_control_indices["test"].append(test_ctrl_indices)

    holdout_indices["validation"] = np.concatenate(holdout_indices["validation"]) if len(holdout_indices["validation"]) > 0 else np.array([], dtype=np.int64)
    holdout_indices["validation"].sort()
    holdout_indices["test"] = np.concatenate(holdout_indices["test"]) if len(holdout_indices["test"]) > 0 else np.array([], dtype=np.int64)
    holdout_indices["test"].sort()
    assert np.isin(holdout_indices["validation"], holdout_indices["test"]).sum() == 0

    mask_val = np.isin(left_indices, holdout_indices["validation"])
    mask_test = np.isin(left_indices, holdout_indices["test"])
    train_indices = left_indices[~(mask_val | mask_test)]

    if dm.data_args.split_control:
        for split in ["validation", "test"]:
            if len(splited_control_indices[split]) > 0:
                splited_control_indices[split] = np.concatenate(splited_control_indices[split])
                splited_control_indices[split].sort()
        mask_val = np.isin(control_indices, splited_control_indices["validation"])
        mask_test = np.isin(control_indices, splited_control_indices["test"])
        splited_control_indices["train"] = control_indices[~(mask_val | mask_test)]
        return {
            "train": np.concatenate([train_indices, splited_control_indices["train"]]),
            "validation": np.concatenate([holdout_indices["validation"], splited_control_indices["validation"]]),
            "test": np.concatenate([holdout_indices["test"], splited_control_indices["test"]]),
        }
    else:
        return {
            "train": np.concatenate([train_indices, control_indices]),
            "validation": np.concatenate([holdout_indices["validation"], control_indices]),
            "test": np.concatenate([holdout_indices["test"], control_indices]),
        }

def split_tahoe100m(dm, cache, target_type, key_info, holdout_setname=None, random_setname=None):
    """
    Split tahoe100m.

    :param dm: Input `dm` value.
    :param cache: Input `cache` value.
    :param target_type: Input `target_type` value.
    :param key_info: Input `key_info` value.
    :param holdout_setname: Input `holdout_setname` value.
    :param random_setname: Input `random_setname` value.
    :return: Computed output(s) for this function.
    """
    assert holdout_setname == "holdout_pert" and random_setname == "random_Perturbseq"

    left_indices = np.arange(cache.n_cells)
    categories = cache.pert_categories
    codes = cache.pert_codes
    control_code = (key_info["control_pert"] == categories).nonzero()[0]
    assert len(control_code) == 1
    control_mask = codes == control_code
    control_indices = left_indices[control_mask]
    left_indices = left_indices[~control_mask]

    holdout_indices = {"validation": [], "test": []}

    if dm.data_args.split_control:
        splited_control_indices = {"train": [], "validation": [], "test": []}
    for tgt_ct_idx, tgt_ct in enumerate(cache.cell_type_categories):
        if tgt_ct not in key_info["holdout_celltype"]:
            continue
        tgt_ct_mask = cache.cell_type_codes == tgt_ct_idx
        if not (tmp_n_cells := np.sum(tgt_ct_mask)):
            continue
        tgt_ct_indices = np.where(tgt_ct_mask)[0]

        n_cells = {"validation": 0, "test": 0}
        for split in dm.all_split_names:
            for tgt_pert in key_info["holdout_pert"][split]:
                if tgt_pert not in set(cache.pert_categories):
                    continue
                tgt_pert_idx = {v: k for k, v in enumerate(cache.pert_categories)}[tgt_pert]

                tgt_pert_mask = cache.pert_codes == tgt_pert_idx
                if not (tmp_n_cells := np.sum(tgt_pert_mask)):
                    continue
                tgt_pert_indices = np.where(tgt_pert_mask)[0]

                shared_indices = tgt_ct_indices[np.isin(tgt_ct_indices, tgt_pert_indices)]
                holdout_indices[split].append(shared_indices)
                n_cells[split] += len(shared_indices)

        if dm.data_args.split_control:
            ctrl_mask_for_ct = cache.pert_codes[tgt_ct_indices] == control_code
            ctrl_indices_for_ct = tgt_ct_indices[ctrl_mask_for_ct]

            n_val, n_test = n_cells["validation"], n_cells["test"]
            N = len(tgt_ct_indices)
            n_train = N - n_val - n_test
            assert n_train > 0

            n_ctrl_val = int(len(ctrl_indices_for_ct) * n_val / N)
            n_ctrl_test = int(len(ctrl_indices_for_ct) * n_test / N)

            rng = np.random.default_rng(dm.seed)
            ctrl_indices_for_ct_shuffled = rng.permutation(ctrl_indices_for_ct)
            val_ctrl_indices = ctrl_indices_for_ct_shuffled[:n_ctrl_val]
            test_ctrl_indices = ctrl_indices_for_ct_shuffled[n_ctrl_val:n_ctrl_val + n_ctrl_test]

            splited_control_indices["validation"].append(val_ctrl_indices)
            splited_control_indices["test"].append(test_ctrl_indices)

    holdout_indices["validation"] = np.concatenate(holdout_indices["validation"]) if len(holdout_indices["validation"]) > 0 else np.array([], dtype=np.int64)
    holdout_indices["validation"].sort()
    holdout_indices["test"] = np.concatenate(holdout_indices["test"]) if len(holdout_indices["test"]) > 0 else np.array([], dtype=np.int64)
    holdout_indices["test"].sort()
    assert np.isin(holdout_indices["validation"], holdout_indices["test"]).sum() == 0

    mask_val = np.isin(left_indices, holdout_indices["validation"])
    mask_test = np.isin(left_indices, holdout_indices["test"])
    train_indices = left_indices[~(mask_val | mask_test)]

    if dm.data_args.split_control:
        for split in ["validation", "test"]:
            if len(splited_control_indices[split]) > 0:
                splited_control_indices[split] = np.concatenate(splited_control_indices[split])
                splited_control_indices[split].sort()
        mask_val = np.isin(control_indices, splited_control_indices["validation"])
        mask_test = np.isin(control_indices, splited_control_indices["test"])
        splited_control_indices["train"] = control_indices[~(mask_val | mask_test)]
        return {
            "train": np.concatenate([train_indices, splited_control_indices["train"]]),
            "validation": np.concatenate([holdout_indices["validation"], splited_control_indices["validation"]]),
            "test": np.concatenate([holdout_indices["test"], splited_control_indices["test"]]),
        }
    else:
        return {
            "train": np.concatenate([train_indices, control_indices]),
            "validation": np.concatenate([holdout_indices["validation"], control_indices]),
            "test": np.concatenate([holdout_indices["test"], control_indices]),
        }

def split_cellxgene(dm, cache, target_type, key_info, holdout_setname=None, random_setname=None):
    """
    Split cellxgene.

    :param dm: Input `dm` value.
    :param cache: Input `cache` value.
    :param target_type: Input `target_type` value.
    :param key_info: Input `key_info` value.
    :param holdout_setname: Input `holdout_setname` value.
    :param random_setname: Input `random_setname` value.
    :return: Computed output(s) for this function.
    """
    assert holdout_setname == "holdout_celltype" and random_setname == "random_RNAseq"

    left_indices = np.arange(cache.n_cells)

    holdout_indices = {"validation": [], "test": []}

    if len(key_info["holdout_celltype"]) == 0:
        n_cells = {"validation": 0, "test": 0}
        for split in dm.all_split_names:
            for tgt_batch in key_info["holdout_batches"][split]:
                if tgt_batch not in set(cache.batch_categories):
                    continue
                tgt_batch_idx = {v: k for k, v in enumerate(cache.batch_categories)}[tgt_batch]

                tgt_batch_mask = cache.batch_codes == tgt_batch_idx
                if not (tmp_n_cells := np.sum(tgt_batch_mask)):
                    continue
                tgt_batch_indices = np.where(tgt_batch_mask)[0]

                holdout_indices[split].append(tgt_batch_indices)
                n_cells[split] += len(tgt_batch_indices)
    else:
        for tgt_ct_idx, tgt_ct in enumerate(cache.cell_type_categories):
            if tgt_ct not in key_info["holdout_celltype"]:
                continue
            tgt_ct_mask = cache.cell_type_codes == tgt_ct_idx
            if not (tmp_n_cells := np.sum(tgt_ct_mask)):
                continue
            tgt_ct_indices = np.where(tgt_ct_mask)[0]

            n_cells = {"validation": 0, "test": 0}
            for split in dm.all_split_names:
                for tgt_batch in key_info["holdout_batches"][split]:
                    if tgt_batch not in set(cache.batch_categories):
                        continue
                    tgt_batch_idx = {v: k for k, v in enumerate(cache.batch_categories)}[tgt_batch]

                    tgt_batch_mask = cache.batch_codes == tgt_batch_idx
                    if not (tmp_n_cells := np.sum(tgt_batch_mask)):
                        continue
                    tgt_batch_indices = np.where(tgt_batch_mask)[0]

                    shared_indices = tgt_ct_indices[np.isin(tgt_ct_indices, tgt_batch_indices)]
                    holdout_indices[split].append(shared_indices)
                    n_cells[split] += len(shared_indices)

    holdout_indices["validation"] = np.concatenate(holdout_indices["validation"]) if len(holdout_indices["validation"]) > 0 else np.array([], dtype=np.int64)
    holdout_indices["validation"].sort()
    holdout_indices["test"] = np.concatenate(holdout_indices["test"]) if len(holdout_indices["test"]) > 0 else np.array([], dtype=np.int64)
    holdout_indices["test"].sort()
    assert np.isin(holdout_indices["validation"], holdout_indices["test"]).sum() == 0

    mask_val = np.isin(left_indices, holdout_indices["validation"])
    mask_test = np.isin(left_indices, holdout_indices["test"])
    train_indices = left_indices[~(mask_val | mask_test)]

    return {
        "train": train_indices,
        "validation": holdout_indices["validation"],
        "test": holdout_indices["test"],
    }

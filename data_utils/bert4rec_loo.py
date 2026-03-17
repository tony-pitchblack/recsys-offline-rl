from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.distributed import barrier, is_distributed, is_rank0


def load_or_build_bert4rec_splits(
    *,
    n_rows: int,
    eligible_val_idx: np.ndarray,
    eligible_test_idx: np.ndarray | None = None,
    val_samples_num: int,
    test_samples_num: int,
    seed: int,
    splits_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits_path = Path(splits_path)
    if is_distributed():
        if not is_rank0():
            barrier()
            z = np.load(str(splits_path))
            return z["train_idx"], z["val_idx"], z["test_idx"]
        if splits_path.exists():
            z = np.load(str(splits_path))
            barrier()
            return z["train_idx"], z["val_idx"], z["test_idx"]
    else:
        if splits_path.exists():
            z = np.load(str(splits_path))
            return z["train_idx"], z["val_idx"], z["test_idx"]

    n_rows = int(n_rows)
    eligible_val_idx = np.asarray(eligible_val_idx, dtype=np.int64)
    if eligible_val_idx.ndim != 1:
        raise ValueError("eligible_val_idx must be 1D")
    if eligible_val_idx.size == 0:
        raise ValueError("No eligible validation sequences for bert4rec_loo splits")

    eligible_test_idx = eligible_val_idx if eligible_test_idx is None else np.asarray(eligible_test_idx, dtype=np.int64)
    if eligible_test_idx.ndim != 1:
        raise ValueError("eligible_test_idx must be 1D")
    if eligible_test_idx.size == 0:
        raise ValueError("No eligible test sequences for bert4rec_loo splits")

    eligible_val_idx = np.unique(eligible_val_idx)
    eligible_test_idx = np.unique(eligible_test_idx)

    val_samples_num = int(val_samples_num)
    test_samples_num = int(test_samples_num)
    if val_samples_num <= 0 or test_samples_num <= 0:
        raise ValueError("val_samples_num and test_samples_num must be > 0 for bert4rec_loo")

    rng = np.random.default_rng(int(seed))
    if int(test_samples_num) > int(eligible_test_idx.size):
        raise ValueError(f"test_samples_num={test_samples_num} exceeds eligible_test={int(eligible_test_idx.size)}")
    test_idx = rng.choice(eligible_test_idx, size=int(test_samples_num), replace=False)

    remaining_val = np.setdiff1d(eligible_val_idx, test_idx, assume_unique=False)
    if int(val_samples_num) > int(remaining_val.size):
        raise ValueError(f"val_samples_num={val_samples_num} exceeds eligible_val_minus_test={int(remaining_val.size)}")
    val_idx = rng.choice(remaining_val, size=int(val_samples_num), replace=False)

    all_idx = np.arange(n_rows, dtype=np.int64)
    used = np.union1d(val_idx, test_idx)
    train_idx = np.setdiff1d(all_idx, used, assume_unique=False)

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(splits_path), train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    if is_distributed():
        barrier()
    return train_idx, val_idx, test_idx


def prepare_sessions_bert4rec_loo(
    *,
    data_directory: str,
    split_df_names: list[str],
    seed: int,
    val_samples_num: int,
    test_samples_num: int,
    rating_threshold: float = 3.5,
    rating_col: str = "rating",
    limit_chunks_pct: float | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    dfs = [pd.read_pickle(str(Path(data_directory) / n)) for n in list(split_df_names)]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    groups = df.groupby("session_id", sort=False)
    items_list: list[torch.Tensor] = []
    signal_list: list[torch.Tensor] = []
    use_float_signal = False
    has_is_buy = "is_buy" in df.columns
    has_rating = str(rating_col) in df.columns
    if not has_is_buy and not has_rating:
        raise KeyError(
            f"Expected either 'is_buy' or '{rating_col}' column in bert4rec_loo dataframe"
        )
    threshold = float(rating_threshold)
    for _, group in groups:
        if "timestamp" in group.columns:
            group = group.sort_values("timestamp", kind="mergesort")
        items = torch.from_numpy(group["item_id"].to_numpy(dtype=np.int64, copy=True))
        if int(items.numel()) == 0:
            continue
        if has_is_buy:
            signal = torch.from_numpy(group["is_buy"].to_numpy(dtype=np.int64, copy=True))
        else:
            ratings = torch.from_numpy(group[str(rating_col)].to_numpy(dtype=np.float32, copy=True))
            signal = (ratings > threshold).to(torch.float32)
            use_float_signal = True
        items_list.append(items)
        signal_list.append(signal)
    splits_root = Path(data_directory)
    if limit_chunks_pct is not None:
        if not (0.0 < float(limit_chunks_pct) <= 1.0):
            raise ValueError("limit_chunks_pct must be in (0, 1]")
        total = int(len(items_list))
        if total <= 0:
            raise ValueError("No sessions found")
        n_keep = max(1, min(total, int(math.ceil(float(total) * float(limit_chunks_pct)))))
        splits_root = splits_root / f"limit_chunks={int(n_keep)}"
        items_list = list(items_list[: int(n_keep)])
        signal_list = list(signal_list[: int(n_keep)])

    eligible = np.asarray([i for i, x in enumerate(items_list) if int(x.numel()) >= 3], dtype=np.int64)
    splits_path = splits_root / "bert4rec_eval" / "dataset_splits.npz"
    train_idx, val_idx, test_idx = load_or_build_bert4rec_splits(
        n_rows=int(len(items_list)),
        eligible_val_idx=eligible,
        eligible_test_idx=eligible,
        val_samples_num=int(val_samples_num),
        test_samples_num=int(test_samples_num),
        seed=int(seed),
        splits_path=splits_path,
    )
    _ = train_idx

    val_mask = np.zeros((int(len(items_list)),), dtype=np.bool_)
    val_mask[np.asarray(val_idx, dtype=np.int64)] = True

    class _Train(Dataset):
        def __len__(self):
            return int(len(items_list))

        def __getitem__(self, idx: int):
            items = items_list[int(idx)]
            signal = signal_list[int(idx)]
            n_drop = 2 if bool(val_mask[int(idx)]) else 1
            if int(items.numel()) <= int(n_drop):
                sig_dtype = torch.float32 if bool(use_float_signal) else torch.long
                return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=sig_dtype)
            return items[: -int(n_drop)], signal[: -int(n_drop)]

    class _Eval(Dataset):
        def __init__(self, indices: np.ndarray, drop_last: int):
            self.indices = np.asarray(indices, dtype=np.int64)
            self.drop_last = int(drop_last)

        def __len__(self):
            return int(self.indices.shape[0])

        def __getitem__(self, i: int):
            idx = int(self.indices[int(i)])
            items = items_list[idx]
            signal = signal_list[idx]
            if int(self.drop_last) > 0:
                items = items[: -int(self.drop_last)]
                signal = signal[: -int(self.drop_last)]
            return items, signal

    train_ds = _Train()
    val_ds = _Eval(val_idx, drop_last=1)
    test_ds = _Eval(test_idx, drop_last=0)
    return train_ds, val_ds, test_ds


__all__ = [
    "load_or_build_bert4rec_splits",
    "prepare_sessions_bert4rec_loo",
]


from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..distributed import barrier, is_distributed, is_rank0

ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _download_if_missing(*, zip_path: Path) -> None:
    if zip_path.exists():
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(ML_1M_URL, str(zip_path))


def _ensure_raw_ratings(*, raw_dir: Path, zip_path: Path) -> Path:
    ratings_path = raw_dir / "ratings.dat"
    if ratings_path.exists():
        return ratings_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(raw_dir))

    nested_dir = raw_dir / "ml-1m"
    nested_ratings = nested_dir / "ratings.dat"
    if ratings_path.exists():
        return ratings_path
    if not nested_ratings.exists():
        raise FileNotFoundError(f"ratings.dat not found after extraction in: {raw_dir}")

    for p in nested_dir.iterdir():
        target = raw_dir / p.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(str(target))
            else:
                target.unlink()
        shutil.move(str(p), str(target))
    shutil.rmtree(str(nested_dir), ignore_errors=True)
    return ratings_path


def _iterative_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    while True:
        n0 = int(len(out))
        item_counts = out["item_id"].value_counts()
        out = out[out["item_id"].map(item_counts) > 2]
        session_counts = out["session_id"].value_counts()
        out = out[out["session_id"].map(session_counts) > 2]
        if int(len(out)) == n0:
            break
    return out


def _build_pop_dict(df: pd.DataFrame, *, item_num: int) -> dict[int, float]:
    counts = np.bincount(df["item_id"].to_numpy(dtype=np.int64, copy=False), minlength=int(item_num)).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return {int(i): 0.0 for i in range(int(item_num))}
    return {int(i): float(counts[i] / total) for i in range(int(item_num))}


def prepare_ml_1m_artifacts(
    *,
    dataset_root: Path,
    data_rel: str,
    seed: int,
    state_size: int = 10,
) -> tuple[str, Path, Path]:
    logger = logging.getLogger(__name__)
    data_dir = (Path(dataset_root) / str(data_rel)).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "sampled_train.df"
    val_path = data_dir / "sampled_val.df"
    test_path = data_dir / "sampled_test.df"
    data_statis_path = data_dir / "data_statis.df"
    pop_dict_path = data_dir / "pop_dict.txt"
    ready = train_path.exists() and val_path.exists() and test_path.exists() and data_statis_path.exists() and pop_dict_path.exists()
    if ready:
        return str(data_dir), data_statis_path, pop_dict_path

    if is_distributed() and (not is_rank0()):
        barrier()
        return str(data_dir), data_statis_path, pop_dict_path

    zip_path = data_dir.parent / "ml-1m.zip"
    raw_dir = data_dir / "raw"
    _download_if_missing(zip_path=zip_path)
    ratings_path = _ensure_raw_ratings(raw_dir=raw_dir, zip_path=zip_path)
    logger.info("ml_1m: using ratings source at %s", str(ratings_path))

    df = pd.read_csv(
        str(ratings_path),
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    df["session_id"] = df["userId"].astype(np.int64)
    df["item_id"] = df["movieId"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df[["session_id", "item_id", "rating", "timestamp"]].sort_values(["session_id", "timestamp"], kind="mergesort")
    df = _iterative_filter(df)

    item_codes, _ = pd.factorize(df["item_id"], sort=True)
    df["item_id"] = item_codes.astype(np.int64)

    session_ids = df["session_id"].drop_duplicates().to_numpy(dtype=np.int64, copy=True)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(session_ids)
    n_sessions = int(session_ids.shape[0])
    n_train = int(0.8 * n_sessions)
    n_val = int(0.1 * n_sessions)
    train_ids = set(session_ids[:n_train].tolist())
    val_ids = set(session_ids[n_train : n_train + n_val].tolist())

    train_df = df[df["session_id"].isin(train_ids)].copy()
    val_df = df[df["session_id"].isin(val_ids)].copy()
    test_df = df[(~df["session_id"].isin(train_ids)) & (~df["session_id"].isin(val_ids))].copy()

    train_df.to_pickle(str(train_path))
    val_df.to_pickle(str(val_path))
    test_df.to_pickle(str(test_path))

    item_num = int(df["item_id"].max()) + 1 if len(df) > 0 else 0
    pd.DataFrame({"state_size": [int(state_size)], "item_num": [int(item_num)]}).to_pickle(str(data_statis_path))

    pop_dict = _build_pop_dict(train_df, item_num=int(item_num))
    with open(pop_dict_path, "w") as f:
        f.write(str(pop_dict))

    logger.info(
        "ml_1m: built artifacts at %s (sessions train/val/test=%d/%d/%d, item_num=%d)",
        str(data_dir),
        int(train_df["session_id"].nunique()),
        int(val_df["session_id"].nunique()),
        int(test_df["session_id"].nunique()),
        int(item_num),
    )

    if is_distributed():
        barrier()
    return str(data_dir), data_statis_path, pop_dict_path


__all__ = ["prepare_ml_1m_artifacts"]

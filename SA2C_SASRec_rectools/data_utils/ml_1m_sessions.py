from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ML1MSessionDataset(Dataset):
    def __init__(self, data_directory: str, split_df_name: str, *, rating_threshold: float, rating_col: str = "rating"):
        super().__init__()
        df = pd.read_pickle(os.path.join(data_directory, split_df_name))
        if "session_id" not in df.columns or "item_id" not in df.columns:
            raise KeyError("ML-1M split is missing required columns: session_id and/or item_id")
        if str(rating_col) not in df.columns:
            raise KeyError(f"ML-1M split is missing rating column: {rating_col}")

        groups = df.groupby("session_id", sort=False)
        items_list = []
        rewards_list = []
        threshold = float(rating_threshold)
        for _, group in groups:
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp", kind="mergesort")
            items = torch.from_numpy(group["item_id"].to_numpy(dtype=np.int64, copy=True))
            ratings = torch.from_numpy(group[str(rating_col)].to_numpy(dtype=np.float32, copy=True))
            if int(items.numel()) == 0:
                continue
            rewards = (ratings > threshold).to(torch.float32)
            items_list.append(items)
            rewards_list.append(rewards)

        self.items_list = items_list
        self.rewards_list = rewards_list

    def __len__(self):
        return int(len(self.items_list))

    def __getitem__(self, idx: int):
        return self.items_list[int(idx)], self.rewards_list[int(idx)]


def collate_ml1m_sessions(batch, pad_item: int):
    items_list, rewards_list = zip(*batch)
    lengths = torch.as_tensor([int(x.numel()) for x in items_list], dtype=torch.long)
    lmax = int(lengths.max().item()) if lengths.numel() > 0 else 0
    bsz = int(len(items_list))
    items_pad = torch.full((bsz, lmax), int(pad_item), dtype=torch.long)
    rewards_pad = torch.zeros((bsz, lmax), dtype=torch.float32)
    for i, (items, rewards) in enumerate(zip(items_list, rewards_list)):
        n = int(items.numel())
        if n == 0:
            continue
        items_pad[i, :n] = items
        rewards_pad[i, :n] = rewards
    return items_pad, rewards_pad, lengths


def make_ml1m_loader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    pad_item: int,
    shuffle: bool,
    sampler=None,
):
    num_workers = max(1, int(num_workers))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=True,
        drop_last=False,
        collate_fn=lambda b: collate_ml1m_sessions(b, pad_item=int(pad_item)),
    )


def make_shifted_batch_from_rewards(
    items_pad: torch.Tensor,
    rewards_pad: torch.Tensor,
    lengths: torch.Tensor,
    *,
    state_size: int,
    old_pad_item: int,
    purchase_only: bool,
    target_mode: str = "multi_position",
):
    bsz, lmax = items_pad.shape
    s = int(state_size)
    if lmax == 0:
        return None

    pos = torch.arange(s, device=items_pad.device).unsqueeze(0).expand(bsz, s)
    base = (lengths - s).unsqueeze(1)
    idx = base + pos
    valid_idx = idx >= 0
    idx = idx.clamp(min=0, max=lmax - 1)

    actions_raw = items_pad.gather(1, idx)
    rewards_raw = rewards_pad.gather(1, idx)
    actions_raw = actions_raw.masked_fill(~valid_idx, int(old_pad_item))
    rewards_raw = rewards_raw.masked_fill(~valid_idx, 0.0)

    actions = torch.where(actions_raw == int(old_pad_item), torch.zeros_like(actions_raw), actions_raw + 1).to(torch.long)
    reward = rewards_raw.to(torch.float32)

    states_x = torch.zeros((bsz, s), dtype=torch.long, device=actions.device)
    states_x[:, 1:] = actions[:, :-1]

    valid_mask = actions != 0
    if bool(purchase_only):
        valid_mask = valid_mask & (reward > 0.0)
    if not bool(valid_mask.any()):
        return None

    keep = valid_mask.sum(dim=1) > 0
    if not bool(keep.all()):
        actions = actions[keep]
        states_x = states_x[keep]
        reward = reward[keep]
        valid_mask = valid_mask[keep]

    idx_positions = torch.arange(s, device=actions.device, dtype=torch.long).unsqueeze(0).expand_as(valid_mask)
    last_idx = (idx_positions * valid_mask.to(torch.long)).amax(dim=1)
    done_mask = torch.zeros_like(valid_mask)
    done_mask[torch.arange(int(actions.shape[0]), device=actions.device), last_idx] = True

    mode = str(target_mode).strip().lower()
    if mode == "one_step":
        one_step_mask = torch.zeros_like(valid_mask)
        one_step_mask[torch.arange(int(actions.shape[0]), device=actions.device), last_idx] = True
        valid_mask = one_step_mask
    elif mode != "multi_position":
        raise ValueError("target_mode must be one of: one_step | multi_position")

    return {
        "states_x": states_x,
        "actions": actions,
        "reward": reward,
        "valid_mask": valid_mask,
        "done_mask": done_mask,
    }


__all__ = [
    "ML1MSessionDataset",
    "collate_ml1m_sessions",
    "make_ml1m_loader",
    "make_shifted_batch_from_rewards",
]

"""Train multimodal ADNI MRI-T1 + PET classifier with OT alignment.

Pipeline overview:
1) Resolve paired MRI/PET files and labels from match CSV.
2) Split by subject (train/val/test), then rebalance class counts where possible.
3) Encode MRI/PET, align PET features to MRI with OT, and classify fused features.
4) Track metrics, save best checkpoint by val macro-F1, and export reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_ENCODER_BACKBONE = "brainmvp"
DEFAULT_MRI_BRAINMVP_CKPT = "results/adni3_mrit1_training_experiments/brainmvp_uniformer/{task}/best_model.pt"
DEFAULT_PET_BRAINMVP_CKPT = "results/adni3_pet_training_experiments/brainmvp_uniformer/{task}/best_model.pt"
PHASE_TO_CSV_PHASE_NAME = {
    "ADNI1": "ADNI 1",
    "ADNI2": "ADNI 2",
    "ADNI3": "ADNI 3",
    "ADNI4": "ADNI 4",
}
TASK_TO_CLASS_TO_LABEL = {
    "binary_ad_cn": {"CN": 0, "AD": 1},
    "ad_mci_cn": {"CN": 0, "MCI": 1, "AD": 2},
}
CLASS_TO_LABEL = TASK_TO_CLASS_TO_LABEL["binary_ad_cn"].copy()
NUM_CLASSES = len(CLASS_TO_LABEL)
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}
CLASS_NAMES = [LABEL_TO_CLASS[idx] for idx in range(NUM_CLASSES)]


def configure_label_space(task: str) -> None:
    """Update global class mappings for the selected classification task."""
    global CLASS_TO_LABEL, NUM_CLASSES, LABEL_TO_CLASS, CLASS_NAMES
    if task not in TASK_TO_CLASS_TO_LABEL:
        raise ValueError(f"Unknown task: {task}")
    CLASS_TO_LABEL = TASK_TO_CLASS_TO_LABEL[task].copy()
    NUM_CLASSES = len(CLASS_TO_LABEL)
    LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}
    CLASS_NAMES = [LABEL_TO_CLASS[idx] for idx in range(NUM_CLASSES)]


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch for reproducible splits and training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multimodal MRI+PET ADNI classifier with OT feature alignment."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary_ad_cn", "ad_mci_cn", "all"],
        default="binary_ad_cn",
        help="Classification task. Use `all` to run both sequentially.",
    )
    parser.add_argument(
        "--adni-phase",
        type=str,
        choices=["ADNI1", "ADNI2", "ADNI3", "ADNI4"],
    )
    parser.add_argument("--match-csv", type=str, default="")
    parser.add_argument("--mri-root", type=str, default="")
    parser.add_argument("--pet-root", type=str, default="")
    parser.add_argument(
        "--samples-csv", type=str, default="",
        help=(
            "Pre-resolved manifest CSV with columns: Subject_ID, Research_Group, "
            "mri_path, pet_path. If provided, --match-csv/--mri-root/--pet-root are ignored."
        ),
    )
    parser.add_argument(
        "--subject-split-csv", type=str, default="",
        help=(
            "Shared split CSV with columns Subject_ID, split (train|val|test). "
            "If provided, overrides internal split generation."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="results/adni_mrit1_pet_ot")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument(
        "--encoder-backbone",
        type=str,
        choices=["small3d", "brainmvp"],
        default=DEFAULT_ENCODER_BACKBONE,
        help=(
            "Feature extractor backbone. `brainmvp` loads trained MRI/PET BrainMVP "
            "checkpoints and uses them for feature extraction before OT."
        ),
    )
    parser.add_argument(
        "--mri-brainmvp-ckpt",
        type=str,
        default=DEFAULT_MRI_BRAINMVP_CKPT,
        help=(
            "MRI BrainMVP checkpoint path. Supports `{task}` placeholder. "
            "If missing, script tries auto-discovery in results/."
        ),
    )
    parser.add_argument(
        "--pet-brainmvp-ckpt",
        type=str,
        default=DEFAULT_PET_BRAINMVP_CKPT,
        help=(
            "PET BrainMVP checkpoint path. Supports `{task}` placeholder. "
            "If missing, script tries auto-discovery in results/."
        ),
    )
    parser.add_argument(
        "--brainmvp-trainable",
        type=lambda x: x.lower() not in ("0", "false", "no"),
        default=True,
        metavar="BOOL",
        help="Fine-tune BrainMVP backbones (default: True). Pass false/0/no to freeze.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help=(
            "LoRA rank for BrainMVP backbone adapters (default: 8). "
            "Set to 0 to disable LoRA and do full fine-tuning when --brainmvp-trainable is true."
        ),
    )
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA scaling factor alpha (default: 16).")
    parser.add_argument(
        "--lora-target",
        type=str,
        default="attn.qkv,attn.proj",
        help=(
            "Comma-separated list of linear layer name suffixes to apply LoRA to "
            "(default: 'attn.qkv,attn.proj'). Add 'mlp.fc1,mlp.fc2' for MLP layers too."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Execution device. Use `cuda` to require GPU.",
    )
    parser.add_argument("--ot-weight", type=float, default=0.1)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iters", type=int, default=40)
    parser.add_argument(
        "--classwise-ot",
        type=lambda x: x.lower() not in ("0", "false", "no"),
        default=True,
        metavar="BOOL",
        help=(
            "Compute OT transport separately per class (default: True). "
            "When True, MRI/PET samples of the same diagnosis are aligned together, "
            "preventing cross-class mixing. Falls back to batch-level OT when a class "
            "has fewer than 2 paired samples in the batch."
        ),
    )
    parser.add_argument(
        "--confusion-matrix-interval",
        type=int,
        default=1,
        help="Save train/val confusion matrix every N epochs (values <=0 are treated as 1).",
    )
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def default_raw_mri_root_for_phase(phase: str) -> str:
    return f"datasets/{phase}/MRIT1/ADNI"


def default_raw_pet_root_for_phase(phase: str) -> str:
    return f"datasets/{phase}/PET/ADNI"


def _latest_existing_path(candidates: List[Path]) -> Optional[Path]:
    """Return newest existing file path from candidates, or None."""
    existing = [path.resolve() for path in candidates if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def resolve_brainmvp_checkpoint(
    checkpoint_arg: str,
    modality: str,
    task: str,
    adni_phase: str,
) -> str:
    """Resolve BrainMVP checkpoint path from explicit arg or known result patterns."""
    text = str(checkpoint_arg).strip()
    if text:
        try:
            text = text.format(task=task, adni_phase=adni_phase.lower())
        except Exception:
            # Keep raw text if it is not a format-template style string.
            pass
        explicit = Path(text).expanduser().resolve()
        if explicit.is_file():
            return str(explicit)

    phase = str(adni_phase).strip().lower()
    mod = str(modality).strip().lower()
    if mod not in {"mri", "pet"}:
        raise ValueError(f"Unknown modality for BrainMVP checkpoint resolution: {modality}")

    modality_token = "mrit1" if mod == "mri" else "pet"
    phase_root = f"{phase}_{modality_token}_training_experiments"
    preferred_patterns = [
        f"results/{phase_root}/brainmvp_uniformer/{task}/best_model.pt",
        f"results/{phase_root}/brainmvp_uniformer/{task}/*/best_model.pt",
        f"results/adni_{modality_token}_brainmvp_uniformer/{task}/best_model.pt",
        f"results/adni_{modality_token}_brainmvp_uniformer/{task}/*/best_model.pt",
    ]
    preferred_candidates: List[Path] = []
    for pattern in preferred_patterns:
        preferred_candidates.extend(Path(".").glob(pattern))
    selected = _latest_existing_path(preferred_candidates)
    if selected is not None:
        return str(selected)

    fallback_candidates = [
        path
        for path in Path(".").glob(f"results/**/brainmvp_uniformer/{task}/**/best_model.pt")
        if modality_token in str(path).lower()
    ]
    selected = _latest_existing_path(fallback_candidates)
    if selected is not None:
        return str(selected)

    raise FileNotFoundError(
        f"Could not resolve {mod.upper()} BrainMVP checkpoint for task `{task}` "
        f"(arg=`{checkpoint_arg}`). Set "
        f"`--{mod}-brainmvp-ckpt /absolute/path/to/best_model.pt` explicitly."
    )


def resolve_existing_root(path: str, label: str) -> str:
    """Validate directory existence and return absolute path string."""
    expanded = Path(path).expanduser().resolve()
    if not expanded.is_dir():
        raise FileNotFoundError(f"{label} root not found: {expanded}")
    return str(expanded)


def build_label(research_group: object) -> Optional[int]:
    """Map raw diagnosis string to current task label index."""
    if pd.isna(research_group):
        return None
    rg = str(research_group).strip().upper()
    return CLASS_TO_LABEL.get(rg)


def map_mri_series_dir_to_preprocessed_path(
    series_dir: str,
    preprocessed_mri_root: str,
    adni_phase: str,
) -> Path:
    """Map raw MRI series directory to expected preprocessed NIfTI path."""
    series_path = Path(series_dir).expanduser().resolve()
    preproc_root = Path(preprocessed_mri_root).expanduser().resolve()
    raw_root = Path(default_raw_mri_root_for_phase(adni_phase)).expanduser().resolve()

    rel = None
    try:
        rel = series_path.relative_to(raw_root)
    except Exception:
        parts = series_path.parts
        for idx in range(len(parts) - 1):
            if parts[idx].upper() == "MRIT1" and parts[idx + 1].upper() == "ADNI":
                rel = Path(*parts[idx + 2 :])
                break
    if rel is None:
        rel = Path(series_path.name)

    out_parent = preproc_root / rel.parent
    for suffix in (f"{series_path.name}_MRI_T1_128.nii.gz", f"{series_path.name}_preproc.nii.gz"):
        candidate = out_parent / suffix
        if candidate.is_file():
            return candidate
    # Return the FreeSurfer name as the default expectation so missing-file
    # reporting points to the right path.
    return out_parent / f"{series_path.name}_MRI_T1_128.nii.gz"


def map_pet_series_dir_to_preprocessed_path(
    series_dir: str,
    preprocessed_pet_root: str,
    adni_phase: str,
) -> Path:
    """Map raw PET series directory to expected preprocessed NIfTI path."""
    series_path = Path(series_dir).expanduser().resolve()
    preproc_root = Path(preprocessed_pet_root).expanduser().resolve()
    raw_root = Path(default_raw_pet_root_for_phase(adni_phase)).expanduser().resolve()

    rel = None
    try:
        rel = series_path.relative_to(raw_root)
    except Exception:
        parts = series_path.parts
        for idx in range(len(parts) - 1):
            if parts[idx].upper() == "PET" and parts[idx + 1].upper() == "ADNI":
                rel = Path(*parts[idx + 2 :])
                break
    if rel is None:
        rel = Path(series_path.name)

    out_parent = preproc_root / rel
    return out_parent / f"{series_path.name}_preproc.nii.gz"


def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by trimming and replacing whitespace with underscores."""
    rename_map = {}
    for col in df.columns:
        new_col = re.sub(r"\s+", "_", str(col).strip())
        rename_map[col] = new_col
    return df.rename(columns=rename_map)


def load_adni_mri_pet_samples(
    match_csv: str,
    mri_root: str,
    pet_root: str,
    adni_phase: str,
) -> pd.DataFrame:
    """Load and validate paired MRI/PET rows, returning resolved file-level samples."""
    match_df = pd.read_csv(match_csv)
    match_df = _normalize_colnames(match_df)

    paired_required_cols = {
        "Subject_ID",
        "MRI_Study_Date",
        "PET_Study_Date",
        "MRI_Research_Group",
        "PET_Research_Group",
        "MRI_Series_Dir",
        "PET_Series_Dir",
    }
    if not paired_required_cols.issubset(set(match_df.columns)):
        phase_name = PHASE_TO_CSV_PHASE_NAME.get(adni_phase, adni_phase)
        if "Phase" in match_df.columns:
            match_df = match_df[match_df["Phase"].astype(str).str.strip() == phase_name].copy()
        raise KeyError(
            "Expected paired MRI/PET schema with columns "
            f"{sorted(paired_required_cols)}; got columns like {sorted(match_df.columns)[:10]}..."
        )

    rows = []
    resolution_counts = defaultdict(int)
    unresolved_rows = 0
    unknown_label_rows = 0

    for _, row in match_df.iterrows():
        mri_group = row.get("MRI_Research_Group")
        pet_group = row.get("PET_Research_Group")
        label = build_label(mri_group)
        if label is None:
            label = build_label(pet_group)
        if label is None:
            unknown_label_rows += 1
            continue

        mri_series_dir = str(row.get("MRI_Series_Dir", "")).strip()
        pet_series_dir = str(row.get("PET_Series_Dir", "")).strip()
        if not mri_series_dir or not pet_series_dir:
            unresolved_rows += 1
            resolution_counts["missing_series_dir"] += 1
            continue

        mri_path = map_mri_series_dir_to_preprocessed_path(
            series_dir=mri_series_dir,
            preprocessed_mri_root=mri_root,
            adni_phase=adni_phase,
        )
        pet_path = map_pet_series_dir_to_preprocessed_path(
            series_dir=pet_series_dir,
            preprocessed_pet_root=pet_root,
            adni_phase=adni_phase,
        )
        if not mri_path.is_file() or not pet_path.is_file():
            unresolved_rows += 1
            if not mri_path.is_file():
                resolution_counts["missing_mri_preprocessed"] += 1
            if not pet_path.is_file():
                resolution_counts["missing_pet_preprocessed"] += 1
            continue

        resolution_counts["paired_series_dir_to_preprocessed"] += 1
        rows.append(
            {
                "Subject_ID": str(row["Subject_ID"]),
                "MRI_Date": str(row["MRI_Study_Date"]),
                "PET_Date": str(row["PET_Study_Date"]),
                "MRI_Description": str(row.get("MRI_Description", "")),
                "PET_Description": str(row.get("PET_Description", "")),
                "Research_Group": str(mri_group if not pd.isna(mri_group) else pet_group),
                "label": int(label),
                "label_name": LABEL_TO_CLASS[int(label)],
                "mri_path": str(mri_path),
                "pet_path": str(pet_path),
                "resolution_mode": "paired_series_dir_to_preprocessed",
            }
        )

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError("No MRI+PET paired samples could be resolved from the match CSV.")

    before_dedup = len(samples)
    samples = samples.sort_values(["Subject_ID", "MRI_Date", "PET_Date", "mri_path", "pet_path"]).drop_duplicates(
        subset=["mri_path", "pet_path"],
        keep="first",
    )

    print(f"[Info] Input match rows: {len(match_df)}")
    print(f"[Info] Resolved paired rows before dedup: {before_dedup}")
    print(f"[Info] Paired rows after dedup by path pair: {len(samples)}")
    print(f"[Info] Unresolved rows: {unresolved_rows}")
    print(f"[Info] Unknown-label rows dropped: {unknown_label_rows}")
    print(f"[Info] Resolution mode counts: {dict(resolution_counts)}")
    print("[Info] Label distribution:")
    print(samples["label_name"].value_counts().to_string())
    return samples.reset_index(drop=True)


def maybe_subsample(
    samples: pd.DataFrame,
    max_samples: int,
    seed: int,
    samples_per_class: int = 0,
) -> pd.DataFrame:
    """Optionally downsample while preserving class balance as much as possible."""
    if max_samples > 0 and samples_per_class > 0:
        raise ValueError("Use only one of --max-samples or --samples-per-class, not both.")

    if samples_per_class > 0:
        rng = np.random.default_rng(seed)
        sampled_parts = []
        insufficient: List[str] = []
        for label in range(NUM_CLASSES):
            group_df = samples[samples["label"] == label]
            class_name = LABEL_TO_CLASS.get(int(label), str(label))
            if len(group_df) < samples_per_class:
                insufficient.append(f"{class_name}({len(group_df)}<{samples_per_class})")
                continue
            sampled_idx = rng.choice(group_df.index.to_numpy(), size=samples_per_class, replace=False)
            sampled_parts.append(samples.loc[sampled_idx])
        if insufficient:
            raise ValueError(
                "Not enough samples for balanced per-class sampling: "
                + ", ".join(sorted(insufficient))
            )
        sampled = pd.concat(sampled_parts, ignore_index=True)
        sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return sampled

    if max_samples <= 0 or len(samples) <= max_samples:
        return samples

    rng = np.random.default_rng(seed)
    sampled_parts = []
    for _, group_df in samples.groupby("label"):
        frac = len(group_df) / len(samples)
        take = max(1, int(round(max_samples * frac)))
        take = min(take, len(group_df))
        sampled_idx = rng.choice(group_df.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(samples.loc[sampled_idx])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return sampled.reset_index(drop=True)


def _safe_split_subjects(
    subject_ids: pd.Series,
    labels: pd.Series,
    test_size: float,
    seed: int,
) -> Tuple[pd.Series, pd.Series]:
    """Run stratified subject split, with deterministic non-stratified fallback."""
    try:
        return train_test_split(
            subject_ids,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as exc:
        print(f"[Warn] Stratified split failed ({exc}); falling back to non-stratified split.")
        return train_test_split(
            subject_ids,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def split_by_subject(
    samples: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by Subject_ID to prevent subject leakage across splits."""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    subject_df = samples.groupby("Subject_ID", as_index=False).agg({"label": "first"})
    trainval_subj, test_subj = _safe_split_subjects(
        subject_ids=subject_df["Subject_ID"],
        labels=subject_df["label"],
        test_size=test_ratio,
        seed=seed,
    )

    trainval_df = subject_df[subject_df["Subject_ID"].isin(trainval_subj)].copy()
    val_relative_ratio = val_ratio / (1.0 - test_ratio)
    train_subj, val_subj = _safe_split_subjects(
        subject_ids=trainval_df["Subject_ID"],
        labels=trainval_df["label"],
        test_size=val_relative_ratio,
        seed=seed,
    )

    train_df = samples[samples["Subject_ID"].isin(set(train_subj))].copy()
    val_df = samples[samples["Subject_ID"].isin(set(val_subj))].copy()
    test_df = samples[samples["Subject_ID"].isin(set(test_subj))].copy()
    return train_df, val_df, test_df


def _target_counts_per_class(
    samples: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
) -> Dict[int, Dict[str, int]]:
    """Compute integer per-class targets for train/val/test from desired ratios."""
    train_ratio = 1.0 - val_ratio - test_ratio
    split_order = ["train", "val", "test"]
    target: Dict[int, Dict[str, int]] = {}

    for label in range(NUM_CLASSES):
        n = int((samples["label"] == label).sum())
        raw = {
            "train": n * train_ratio,
            "val": n * val_ratio,
            "test": n * test_ratio,
        }
        base = {k: int(np.floor(v)) for k, v in raw.items()}
        remaining = int(n - sum(base.values()))
        frac_order = sorted(
            split_order,
            key=lambda k: (raw[k] - base[k], -split_order.index(k)),
            reverse=True,
        )
        for idx in range(remaining):
            base[frac_order[idx % len(frac_order)]] += 1
        target[label] = base
    return target


def _current_counts_per_class(
    split_frames: Dict[str, pd.DataFrame],
) -> Dict[int, Dict[str, int]]:
    """Return current class counts for each split frame."""
    current: Dict[int, Dict[str, int]] = {}
    for label in range(NUM_CLASSES):
        current[label] = {
            split_name: int((frame["label"] == label).sum())
            for split_name, frame in split_frames.items()
        }
    return current


def rebalance_split_class_counts_by_subject(
    samples: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Move whole subjects between splits to better match class-count targets."""
    split_subjects: Dict[str, set[str]] = {
        "train": set(train_df["Subject_ID"].astype(str).tolist()),
        "val": set(val_df["Subject_ID"].astype(str).tolist()),
        "test": set(test_df["Subject_ID"].astype(str).tolist()),
    }
    split_frames: Dict[str, pd.DataFrame] = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    subject_label_counts = (
        samples.groupby(["Subject_ID", "label"]).size().unstack(fill_value=0)
        .reindex(columns=list(range(NUM_CLASSES)), fill_value=0)
    )
    subject_total_rows = subject_label_counts.sum(axis=1)
    subject_single_label: Dict[str, Optional[int]] = {}
    for subject_id, row in subject_label_counts.iterrows():
        non_zero_labels = [idx for idx in range(NUM_CLASSES) if int(row.iloc[idx]) > 0]
        if len(non_zero_labels) == 1:
            subject_single_label[str(subject_id)] = int(non_zero_labels[0])
        else:
            subject_single_label[str(subject_id)] = None

    target = _target_counts_per_class(samples=samples, val_ratio=val_ratio, test_ratio=test_ratio)
    current = _current_counts_per_class(split_frames=split_frames)

    # Greedy subject moves: only subjects with a single label are eligible,
    # so we preserve subject integrity and avoid mixed-label ambiguity.
    max_passes = 6
    for _ in range(max_passes):
        moved_any = False
        for label in range(NUM_CLASSES):
            for dst in ("train", "val", "test"):
                deficit = target[label][dst] - current[label][dst]
                if deficit <= 0:
                    continue

                src_candidates = [
                    split_name
                    for split_name in ("train", "val", "test")
                    if split_name != dst and current[label][split_name] > target[label][split_name]
                ]
                src_candidates = sorted(
                    src_candidates,
                    key=lambda split_name: current[label][split_name] - target[label][split_name],
                    reverse=True,
                )
                if not src_candidates:
                    continue

                moved = False
                for src in src_candidates:
                    candidates: List[Tuple[int, str]] = []
                    for subject_id in split_subjects[src]:
                        if subject_single_label.get(subject_id) != label:
                            continue
                        rows = int(subject_total_rows.get(subject_id, 0))
                        if rows <= 0:
                            continue
                        candidates.append((rows, subject_id))

                    if not candidates:
                        continue

                    candidates.sort(key=lambda item: (abs(item[0] - deficit), item[0], item[1]))
                    rows_to_move, chosen_subject = candidates[0]

                    split_subjects[src].remove(chosen_subject)
                    split_subjects[dst].add(chosen_subject)
                    current[label][src] -= rows_to_move
                    current[label][dst] += rows_to_move
                    moved = True
                    moved_any = True
                    break

                if moved:
                    continue
        if not moved_any:
            break

    # Rebuild frames from updated subject assignments.
    rebuilt = {}
    for split_name in ("train", "val", "test"):
        rebuilt[split_name] = samples[samples["Subject_ID"].astype(str).isin(split_subjects[split_name])].copy()

    final_counts = _current_counts_per_class(split_frames=rebuilt)
    unresolved = []
    for label in range(NUM_CLASSES):
        for split_name in ("train", "val", "test"):
            if final_counts[label][split_name] != target[label][split_name]:
                unresolved.append(
                    f"{LABEL_TO_CLASS.get(label, str(label))}:{split_name} "
                    f"{final_counts[label][split_name]}/{target[label][split_name]}"
                )
    if unresolved:
        print(
            "[Warn] Could not perfectly match per-class split targets with subject-level constraints: "
            + ", ".join(unresolved)
        )
    else:
        print("[Info] Subject-level split rebalanced to exact per-class row targets.")

    return rebuilt["train"], rebuilt["val"], rebuilt["test"]


class MRIPEPairDataset(Dataset):
    """Dataset that returns paired (MRI volume, PET volume, label) tensors."""

    def __init__(self, df: pd.DataFrame, target_shape: Tuple[int, int, int], augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        mean = float(volume.mean())
        std = float(volume.std())
        if std < 1e-6:
            return volume - mean
        return (volume - mean) / std

    def _load_one(self, path: str) -> torch.Tensor:
        volume = nib.load(path).get_fdata(dtype=np.float32)
        if volume.ndim == 4:
            volume = volume[..., 0]
        volume = self._normalize(volume)
        # Resize to a fixed shape so batches are stackable regardless of source spacing.
        tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return tensor.squeeze(0)

    @staticmethod
    def _augment_pair(mri: torch.Tensor, pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < 0.5:
            mri = torch.flip(mri, dims=[3])
            pet = torch.flip(pet, dims=[3])
        if torch.rand(1).item() < 0.2:
            mri = torch.flip(mri, dims=[1])
            pet = torch.flip(pet, dims=[1])
        if torch.rand(1).item() < 0.25:
            mri = mri + torch.randn_like(mri) * 0.02
            pet = pet + torch.randn_like(pet) * 0.02
        return mri, pet

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        mri_tensor = self._load_one(row["mri_path"])
        pet_tensor = self._load_one(row["pet_path"])
        if self.augment:
            mri_tensor, pet_tensor = self._augment_pair(mri_tensor, pet_tensor)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return mri_tensor, pet_tensor, label


def _load_uniformer_small():
    """Dynamically load BrainMVP UniFormer implementation from local source."""
    module_path = Path(__file__).resolve().parent / "BrainMVP" / "Downstream" / "model" / "uniformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"BrainMVP UniFormer module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("brainmvp_uniformer_module_for_ot", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load BrainMVP module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.uniformer_small


def _unwrap_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    """Extract model tensor dict from common checkpoint wrapper formats."""
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            maybe = ckpt_obj.get(key)  # type: ignore[arg-type]
            if isinstance(maybe, dict):
                return maybe
        if ckpt_obj and all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj  # type: ignore[return-value]
    raise RuntimeError("Unsupported checkpoint format. Expected dict or dict with `state_dict`.")


def _normalize_backbone_key(key: str) -> Optional[str]:
    """Normalize checkpoint keys to UniFormer backbone keys, skipping non-backbone heads."""
    k = key
    if k.startswith("module."):
        k = k[len("module.") :]
    if k.startswith("encoder.uniformer."):
        k = k[len("encoder.uniformer.") :]
    elif k.startswith("uniformer."):
        k = k[len("uniformer.") :]
    elif k.startswith("encoder."):
        k = k[len("encoder.") :]
    elif k.startswith("backbone."):
        k = k[len("backbone.") :]

    if (
        k.startswith("classifier.")
        or k.startswith("decoder.")
        or k.startswith("rep_template")
        or k.startswith("head.")
    ):
        return None
    return k


class LoRALinear(nn.Module):
    """Frozen base linear layer with trainable low-rank LoRA adapters (Hu et al., 2021).

    The forward pass computes:
        y = base(x) + (x @ A.T @ B.T) * (alpha / rank)

    where A is initialised with Kaiming uniform and B is zero-initialised
    so the adapter output is exactly zero at the start of training.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = float(alpha) / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora_to_backbone(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    target_suffixes: Tuple[str, ...] = ("attn.qkv", "attn.proj"),
) -> int:
    """Replace matching nn.Linear submodules with LoRALinear in-place.

    Only modules whose dotted name ends with one of `target_suffixes` are
    replaced.  The base weights inside each LoRALinear are frozen; only the
    LoRA A/B matrices require gradients.

    Returns the number of replacements made.
    """
    replaced = 0
    for name, module in list(backbone.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suf) for suf in target_suffixes):
            continue
        parts = name.split(".")
        parent: nn.Module = backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(base=module, rank=rank, alpha=alpha))
        replaced += 1
    return replaced


class BrainMVPUniFormerFeatureEncoder(nn.Module):
    """BrainMVP UniFormer encoder with optional projection to requested feature_dim."""

    BACKBONE_DIM = 512

    def __init__(
        self,
        feature_dim: int,
        input_shape: int,
        checkpoint_path: str,
        trainable_backbone: bool,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_target_suffixes: Tuple[str, ...] = ("attn.qkv", "attn.proj"),
    ):
        super().__init__()
        uniformer_small = _load_uniformer_small()
        self.backbone = uniformer_small(img_size=input_shape, in_chans=1)
        self.backbone_trainable = bool(trainable_backbone)
        self.use_lora = self.backbone_trainable and int(lora_rank) > 0
        if int(feature_dim) == self.BACKBONE_DIM:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.BACKBONE_DIM, feature_dim),
                nn.ReLU(inplace=True),
            )

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"BrainMVP checkpoint not found: {checkpoint}")
        self._load_backbone_weights(str(checkpoint))

        if self.use_lora:
            # Freeze all backbone weights, then inject trainable LoRA adapters.
            for param in self.backbone.parameters():
                param.requires_grad = False
            n_replaced = apply_lora_to_backbone(
                self.backbone,
                rank=int(lora_rank),
                alpha=float(lora_alpha),
                target_suffixes=lora_target_suffixes,
            )
            n_lora = sum(
                p.numel() for p in self.backbone.parameters() if p.requires_grad
            )
            n_total = sum(p.numel() for p in self.backbone.parameters())
            print(
                f"[Info] LoRA applied to backbone: {n_replaced} linears replaced "
                f"(rank={lora_rank}, alpha={lora_alpha}). "
                f"Trainable backbone params: {n_lora:,} / {n_total:,} "
                f"({100 * n_lora / max(n_total, 1):.2f}%)"
            )
        else:
            for param in self.backbone.parameters():
                param.requires_grad = self.backbone_trainable
            if not self.backbone_trainable:
                self.backbone.eval()

    def _load_backbone_weights(self, checkpoint_path: str) -> None:
        # Load only compatible backbone tensors and ignore classifier/decoder heads.
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        src_state = _unwrap_state_dict(ckpt)
        dst_state = self.backbone.state_dict()

        loadable: Dict[str, torch.Tensor] = {}
        skipped = 0
        for raw_key, tensor in src_state.items():
            key = _normalize_backbone_key(raw_key)
            if key is None or key not in dst_state:
                skipped += 1
                continue
            if tensor.shape != dst_state[key].shape:
                skipped += 1
                continue
            loadable[key] = tensor

        if not loadable:
            raise RuntimeError(
                f"No compatible backbone tensors found in BrainMVP checkpoint: {checkpoint_path}"
            )
        missing, unexpected = self.backbone.load_state_dict(loadable, strict=False)
        unresolved = len(missing) + len(unexpected)
        print(
            f"[Info] BrainMVP encoder loaded from {checkpoint_path} | "
            f"loaded={len(loadable)}, skipped={skipped}, unresolved_after_load={unresolved}"
        )

    def train(self, mode: bool = True):
        super().train(mode=mode)
        # With LoRA the backbone stays in train mode so its dropout/norm layers
        # behave correctly; only the base linear weights are frozen.
        # Without LoRA and with a frozen backbone, force eval to disable dropout.
        if not self.use_lora and not self.backbone_trainable:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        _, _, _, _, x4 = self.backbone(x)
        pooled = F.adaptive_avg_pool3d(x4, output_size=1).flatten(1)
        z = self.proj(pooled)
        return F.normalize(z, dim=1)


def sinkhorn_plan(cost: torch.Tensor, epsilon: float, num_iters: int) -> torch.Tensor:
    """Compute entropic OT plan between two sample sets using log-domain Sinkhorn.

    This solves:
      min_P <P, C> + epsilon * sum_ij P_ij (log(P_ij) - 1)
    under approximately uniform marginals over rows/columns.

    Inputs:
    - cost: pairwise squared-distance matrix with shape [n_mri, n_pet]
    - epsilon: entropy temperature; larger values produce smoother plans
    - num_iters: Sinkhorn scaling iterations

    Returns:
    - transport plan P with shape [n_mri, n_pet]

    Notes:
    - The computation is performed in log-space (`u`, `v`) to avoid numerical
      underflow/overflow when epsilon is small.
    - Cost is normalized by batch mean so one global epsilon behaves more
      consistently across batches with different raw distance scales.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0 for Sinkhorn.")
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D matrix.")

    n, m = cost.shape
    if n == 0 or m == 0:
        raise ValueError("cost matrix cannot be empty.")

    # Normalize cost scale so fixed epsilon has similar smoothing strength
    # across batches/datasets.
    eps = float(epsilon)
    cost_scaled = cost / (cost.detach().mean().clamp_min(1e-6))
    log_k = -cost_scaled / eps

    # Uniform marginals a_i = 1/n, b_j = 1/m in log-space.
    log_a = -torch.log(torch.tensor(float(n), device=cost.device, dtype=cost.dtype))
    log_b = -torch.log(torch.tensor(float(m), device=cost.device, dtype=cost.dtype))

    # Dual scaling vectors updated by alternating row/column normalization.
    u = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    v = torch.zeros(m, device=cost.device, dtype=cost.dtype)
    for _ in range(max(1, int(num_iters))):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)

    # Recover primal plan from log-domain representation.
    log_p = log_k + u.unsqueeze(1) + v.unsqueeze(0)
    return torch.exp(log_p)



class MRIPEOTClassifier(nn.Module):
    """Dual-encoder classifier with Sinkhorn OT alignment.

    Pipeline:
    1. Encode MRI and PET into latent vectors.
    2. Align PET samples toward MRI samples via log-domain Sinkhorn transport.
    3. Fuse [mri, aligned_pet, elementwise_product] and classify.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        dropout: float,
        ot_epsilon: float,
        ot_iters: int,
        encoder_backbone: str = "brainmvp",
        mri_brainmvp_ckpt: str = "",
        pet_brainmvp_ckpt: str = "",
        brainmvp_trainable: bool = True,
        brainmvp_input_shape: int = 128,
        classwise_ot: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_target_suffixes: Tuple[str, ...] = ("attn.qkv", "attn.proj"),
    ):
        super().__init__()
        self.classwise_ot = bool(classwise_ot)
        self.encoder_backbone = str(encoder_backbone).strip().lower()
        if self.encoder_backbone == "brainmvp":
            if not mri_brainmvp_ckpt or not pet_brainmvp_ckpt:
                raise ValueError(
                    "BrainMVP backbone requires both --mri-brainmvp-ckpt and --pet-brainmvp-ckpt."
                )
            encoder_kwargs = dict(
                feature_dim=feature_dim,
                input_shape=int(brainmvp_input_shape),
                trainable_backbone=bool(brainmvp_trainable),
                lora_rank=int(lora_rank),
                lora_alpha=float(lora_alpha),
                lora_target_suffixes=lora_target_suffixes,
            )
            self.mri_encoder = BrainMVPUniFormerFeatureEncoder(
                checkpoint_path=mri_brainmvp_ckpt, **encoder_kwargs
            )
            self.pet_encoder = BrainMVPUniFormerFeatureEncoder(
                checkpoint_path=pet_brainmvp_ckpt, **encoder_kwargs
            )
        else:
            raise ValueError(f"Unknown encoder backbone: {encoder_backbone}")
        self.ot_epsilon = float(ot_epsilon)
        self.ot_iters = int(ot_iters)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, num_classes),
        )

    def forward(
        self,
        mri: torch.Tensor,
        pet: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one paired forward pass.

        Args:
        - mri: shape [B, 1, D, H, W]
        - pet: shape [B, 1, D, H, W]
        - labels: shape [B] integer class labels (used for classwise OT when available)

        Returns:
        - logits: shape [B, num_classes]
        - ot_loss: scalar transport cost used as alignment regularizer
        """
        mri_feat = self.mri_encoder(mri)
        pet_feat = self.pet_encoder(pet)

        if self.classwise_ot and labels is not None:
            # Per-class transport: align MRI and PET only within the same diagnosis.
            # This prevents cross-class feature mixing (e.g. AD MRI → CN PET).
            # Falls back to batch-level OT for classes with <2 samples in the batch.
            pet_aligned = torch.zeros_like(mri_feat)
            ot_loss = torch.tensor(0.0, device=mri_feat.device, dtype=mri_feat.dtype)
            num_classes_present = 0
            for c in torch.unique(labels):
                mask = labels == c
                if mask.sum() < 2:
                    # Degenerate: copy PET features unchanged for singleton classes.
                    pet_aligned[mask] = pet_feat[mask]
                    continue
                mri_c = mri_feat[mask]
                pet_c = pet_feat[mask]
                cost_c = torch.cdist(mri_c, pet_c, p=2).pow(2)
                transport_c = sinkhorn_plan(
                    cost=cost_c, epsilon=self.ot_epsilon, num_iters=self.ot_iters
                )
                row_mass_c = transport_c.sum(dim=1, keepdim=True).clamp_min(1e-8)
                pet_aligned[mask] = (transport_c @ pet_c) / row_mass_c
                ot_loss = ot_loss + torch.sum(transport_c * cost_c)
                num_classes_present += 1
            if num_classes_present > 0:
                ot_loss = ot_loss / num_classes_present
        else:
            # Batch-level transport: each MRI sample receives a convex combination of
            # all PET samples in the batch (used when labels unavailable or classwise disabled).
            cost = torch.cdist(mri_feat, pet_feat, p=2).pow(2)
            transport = sinkhorn_plan(
                cost=cost, epsilon=self.ot_epsilon, num_iters=self.ot_iters
            )
            row_mass = transport.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pet_aligned = (transport @ pet_feat) / row_mass
            ot_loss = torch.sum(transport * cost)

        # Multiplicative term captures cross-modal agreement per feature dimension.
        fused = torch.cat([mri_feat, pet_aligned, mri_feat * pet_aligned], dim=1)
        logits = self.classifier(fused)
        return logits, ot_loss


def multiclass_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[np.ndarray],
) -> Dict[str, object]:
    """Compute robust multiclass metrics from predictions and probabilities.

    Returned dictionary includes:
    - overall: accuracy, macro precision/recall/F1, balanced accuracy, MCC, AUC
    - per-class: precision/recall/specificity/F1/support
    - confusion matrix (fixed class order = `range(NUM_CLASSES)`)

    The function is defensive:
    - returns NaNs for undefined settings (empty split, missing classes for AUC)
    - avoids divide-by-zero when a class has no positive/negative samples
    """
    if len(y_true) == 0:
        metrics: Dict[str, object] = {
            "accuracy": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "sensitivity_macro": float("nan"),
            "specificity_macro": float("nan"),
            "f1_macro": float("nan"),
            "balanced_accuracy": float("nan"),
            "mcc": float("nan"),
            "auc_roc_ovr_macro": float("nan"),
            "confusion_matrix": [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)],
        }
        for class_name in CLASS_NAMES:
            metrics[f"precision_{class_name}"] = float("nan")
            metrics[f"recall_{class_name}"] = float("nan")
            metrics[f"specificity_{class_name}"] = float("nan")
            metrics[f"f1_{class_name}"] = float("nan")
            metrics[f"support_{class_name}"] = 0.0
        return metrics

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(NUM_CLASSES)))

    accuracy = float((y_true_np == y_pred_np).mean())
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    precision_per_class, recall_per_class, f1_per_class, supports = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )

    # Balanced accuracy here is macro recall over classes that are present in
    # this split (support > 0).
    support_mask = np.asarray(supports, dtype=np.float64) > 0
    if np.any(support_mask):
        bacc = float(np.mean(np.asarray(recall_per_class, dtype=np.float64)[support_mask]))
    else:
        bacc = float("nan")
    try:
        mcc = float(matthews_corrcoef(y_true_np, y_pred_np))
    except Exception:
        mcc = float("nan")

    # Per-class specificity = TN / (TN + FP), computed from one-vs-rest view.
    specificity_per_class: List[float] = []
    total = cm.sum()
    for class_idx in range(NUM_CLASSES):
        tp = float(cm[class_idx, class_idx])
        fp = float(cm[:, class_idx].sum() - tp)
        fn = float(cm[class_idx, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        denom = tn + fp
        specificity_per_class.append(float(tn / denom) if denom > 0 else float("nan"))
    specificity_macro = float(np.nanmean(np.asarray(specificity_per_class, dtype=np.float64)))

    # AUC is only valid when class coverage conditions are satisfied.
    auc_macro = float("nan")
    if y_prob is not None and len(y_prob) == len(y_true_np):
        try:
            if NUM_CLASSES == 2:
                if np.unique(y_true_np).size >= 2:
                    auc_macro = float(roc_auc_score(y_true_np, y_prob[:, 1]))
            else:
                # sklearn multiclass AUC expects all classes in y_true.
                if np.unique(y_true_np).size == NUM_CLASSES:
                    auc_macro = float(roc_auc_score(y_true_np, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc_macro = float("nan")

    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "sensitivity_macro": float(recall_macro),
        "specificity_macro": specificity_macro,
        "f1_macro": float(f1_macro),
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "auc_roc_ovr_macro": auc_macro,
        "confusion_matrix": cm.tolist(),
    }
    for idx, class_name in enumerate(CLASS_NAMES):
        metrics[f"precision_{class_name}"] = float(precision_per_class[idx])
        metrics[f"recall_{class_name}"] = float(recall_per_class[idx])
        metrics[f"specificity_{class_name}"] = float(specificity_per_class[idx])
        metrics[f"f1_{class_name}"] = float(f1_per_class[idx])
        metrics[f"support_{class_name}"] = float(supports[idx])
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    epoch_idx: int,
    total_epochs: int,
    ot_weight: float,
    require_cuda: bool = False,
) -> Tuple[float, float, float, Dict[str, object]]:
    """Run one train or eval epoch and aggregate losses + metrics.

    Behavior:
    - `optimizer is None` -> evaluation mode (`no_grad` not forced, but no step)
    - `optimizer provided` -> training mode with backprop/update

    Objective per batch:
      total_loss = cls_loss + ot_weight * ot_loss

    Returns:
    - avg total loss, avg classification loss, avg OT loss, metrics dict
    """
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_ot_loss = 0.0
    total_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []

    progress = tqdm(
        loader,
        desc=f"{split_name} {epoch_idx}/{total_epochs}",
        leave=False,
    )
    checked_cuda = False
    for mri_vol, pet_vol, labels in progress:
        mri_vol = mri_vol.to(device, non_blocking=True)
        pet_vol = pet_vol.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if require_cuda and not checked_cuda:
            # Fail fast for accidental CPU execution when GPU-only runs are expected.
            model_on_cuda = next(model.parameters()).is_cuda
            batch_on_cuda = bool(mri_vol.is_cuda and pet_vol.is_cuda and labels.is_cuda)
            if not (model_on_cuda and batch_on_cuda):
                raise RuntimeError(
                    "CUDA is required, but model/batch tensors are not on CUDA. "
                    f"model_on_cuda={model_on_cuda}, batch_on_cuda={batch_on_cuda}"
                )
            checked_cuda = True

        if training:
            optimizer.zero_grad(set_to_none=True)

        # Total objective = supervised CE + weighted OT alignment.
        logits, ot_loss = model(mri_vol, pet_vol, labels=labels)
        cls_loss = criterion(logits, labels)
        loss = cls_loss + float(ot_weight) * ot_loss

        if training:
            loss.backward()
            optimizer.step()

        # Weighted accumulation by batch size ensures correct epoch means even
        # if the final batch is smaller.
        batch_n = int(labels.size(0))
        total_count += batch_n
        total_loss += float(loss.item()) * batch_n
        total_cls_loss += float(cls_loss.item()) * batch_n
        total_ot_loss += float(ot_loss.item()) * batch_n

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())
        all_prob.append(probs.detach().cpu().numpy())

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            ot=f"{ot_loss.item():.4f}",
        )

    avg_loss = total_loss / max(total_count, 1)
    avg_cls = total_cls_loss / max(total_count, 1)
    avg_ot = total_ot_loss / max(total_count, 1)
    # `all_prob` is required for AUC; if empty, AUC remains NaN in metrics.
    prob_array = np.concatenate(all_prob, axis=0) if all_prob else None
    metrics = multiclass_metrics(all_true, all_pred, y_prob=prob_array)
    return avg_loss, avg_cls, avg_ot, metrics


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_confusion_matrix(path: str, matrix: List[List[int]], class_names: List[str]) -> None:
    cm = np.asarray(matrix, dtype=np.int64)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(path, index=True)


def save_confusion_matrix_png(path: str, matrix: List[List[int]], class_names: List[str], title: str) -> None:
    import matplotlib.pyplot as plt

    cm = np.asarray(matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_training_plots(history_df: pd.DataFrame, output_dir: str) -> None:
    if history_df.empty:
        return

    import matplotlib.pyplot as plt

    epochs = history_df["epoch"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    axes[0, 0].plot(epochs, history_df["train_loss"], label="train_total", linewidth=2)
    axes[0, 0].plot(epochs, history_df["val_loss"], label="val_total", linewidth=2)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history_df["train_cls_loss"], label="train_cls", linewidth=2)
    axes[0, 1].plot(epochs, history_df["val_cls_loss"], label="val_cls", linewidth=2)
    axes[0, 1].plot(epochs, history_df["train_ot_loss"], label="train_ot", linewidth=2)
    axes[0, 1].plot(epochs, history_df["val_ot_loss"], label="val_ot", linewidth=2)
    axes[0, 1].set_title("Classification / OT Loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history_df["train_accuracy"], label="train_acc", linewidth=2)
    axes[1, 0].plot(epochs, history_df["val_accuracy"], label="val_acc", linewidth=2)
    axes[1, 0].plot(epochs, history_df["train_f1_score"], label="train_f1", linewidth=2)
    axes[1, 0].plot(epochs, history_df["val_f1_score"], label="val_f1", linewidth=2)
    axes[1, 0].set_title("Accuracy / F1")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history_df["train_mcc"], label="train_mcc", linewidth=2)
    axes[1, 1].plot(epochs, history_df["val_mcc"], label="val_mcc", linewidth=2)
    axes[1, 1].plot(epochs, history_df["train_auc_roc_ovr_macro"], label="train_auc", linewidth=2)
    axes[1, 1].plot(epochs, history_df["val_auc_roc_ovr_macro"], label="val_auc", linewidth=2)
    axes[1, 1].set_title("MCC / AUC")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), bbox_inches="tight")
    plt.close(fig)


def selected_metric_view(metrics: Dict[str, object]) -> Dict[str, float]:
    """Return compact metric subset for logs/CSVs/summaries."""
    return {
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "auc_roc": float(metrics.get("auc_roc_ovr_macro", float("nan"))),
        "sensitivity_recall": float(metrics.get("sensitivity_macro", float("nan"))),
        "specificity": float(metrics.get("specificity_macro", float("nan"))),
        "precision": float(metrics.get("precision_macro", float("nan"))),
        "f1_score": float(metrics.get("f1_macro", float("nan"))),
        "mcc": float(metrics.get("mcc", float("nan"))),
    }


def _print_split_label_distribution(split_name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        print(f"[Info] {split_name} label distribution: <empty>")
        return

    if "label_name" in frame.columns:
        counts = frame["label_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    else:
        counts = pd.Series(
            [int((frame["label"] == class_idx).sum()) for class_idx in range(NUM_CLASSES)],
            index=CLASS_NAMES,
            dtype=np.int64,
        )
    print(f"[Info] {split_name} label distribution:")
    print(counts.to_string())


def _load_samples_from_manifest(samples_csv: str, task: str) -> pd.DataFrame:
    """Load a pre-resolved MRI+PET manifest and apply task-specific label filtering."""
    df = pd.read_csv(samples_csv)
    required = {"Subject_ID", "Research_Group", "mri_path", "pet_path"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"--samples-csv is missing required columns: {sorted(missing)}")
    class_map = TASK_TO_CLASS_TO_LABEL[task]
    df = df[df["Research_Group"].str.upper().isin(class_map)].copy()
    df["label"] = df["Research_Group"].str.upper().map(class_map)
    df["label_name"] = df["label"].map({v: k for k, v in class_map.items()})
    return df.reset_index(drop=True)


def _apply_subject_split(samples: pd.DataFrame, subject_split_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split samples using a pre-built subject-split CSV (Subject_ID, split columns)."""
    split_df = pd.read_csv(subject_split_csv)
    if not {"Subject_ID", "split"}.issubset(set(split_df.columns)):
        raise KeyError("--subject-split-csv must have columns: Subject_ID, split")
    split_map = dict(zip(split_df["Subject_ID"].astype(str), split_df["split"]))
    train_df = samples[samples["Subject_ID"].astype(str).map(split_map) == "train"].copy()
    val_df   = samples[samples["Subject_ID"].astype(str).map(split_map) == "val"].copy()
    test_df  = samples[samples["Subject_ID"].astype(str).map(split_map) == "test"].copy()
    return train_df, val_df, test_df


def run_single_task(args: argparse.Namespace) -> None:
    """Run full train/val/test pipeline for one task configuration."""
    adni_phase = str(getattr(args, "adni_phase", "")).strip().upper()
    confusion_matrix_interval = int(args.confusion_matrix_interval)
    if confusion_matrix_interval <= 0:
        print(
            "[Warn] --confusion-matrix-interval <= 0 was provided. "
            "Using 1 to save confusion matrices every epoch."
        )
        confusion_matrix_interval = 1

    configure_label_space(args.task)
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    confusion_dir = os.path.join(args.output_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)

    requested_device = str(args.device).strip().lower()
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` requested, but CUDA is not available in this Python environment.")
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_shape = (args.depth, args.height, args.width)

    print(f"[Info] Device: {device}")
    print(f"[Info] Task: {args.task}")
    print(f"[Info] Classes: {CLASS_NAMES}")
    print(f"[Info] Target volume shape: {target_shape}")
    print(f"[Info] Encoder backbone: {args.encoder_backbone}")
    print(
        f"[Info] OT settings: weight={args.ot_weight}, "
        f"epsilon={args.ot_epsilon}, iters={args.ot_iters}"
    )

    resolved_mri_brainmvp_ckpt = ""
    resolved_pet_brainmvp_ckpt = ""
    if str(args.encoder_backbone).strip().lower() == "brainmvp":
        resolved_mri_brainmvp_ckpt = resolve_brainmvp_checkpoint(
            checkpoint_arg=args.mri_brainmvp_ckpt,
            modality="mri",
            task=args.task,
            adni_phase=adni_phase or "ADNI3",
        )
        resolved_pet_brainmvp_ckpt = resolve_brainmvp_checkpoint(
            checkpoint_arg=args.pet_brainmvp_ckpt,
            modality="pet",
            task=args.task,
            adni_phase=adni_phase or "ADNI3",
        )
        print(f"[Info] MRI BrainMVP checkpoint: {resolved_mri_brainmvp_ckpt}")
        print(f"[Info] PET BrainMVP checkpoint: {resolved_pet_brainmvp_ckpt}")
        print(f"[Info] BrainMVP backbones trainable: {bool(args.brainmvp_trainable)}")
        use_lora = bool(args.brainmvp_trainable) and int(args.lora_rank) > 0
        print(
            f"[Info] LoRA: {'enabled' if use_lora else 'disabled'}"
            + (f" (rank={args.lora_rank}, alpha={args.lora_alpha}, targets={args.lora_target})" if use_lora else "")
        )
        print(f"[Info] Classwise OT: {bool(args.classwise_ot)}")

    # ── Load samples ─────────────────────────────────────────────────────────
    samples_csv = str(getattr(args, "samples_csv", "")).strip()
    if samples_csv:
        print(f"[Info] Loading samples from --samples-csv: {samples_csv}")
        samples = _load_samples_from_manifest(samples_csv, args.task)
    else:
        match_csv = str(Path(args.match_csv).expanduser().resolve())
        mri_root  = resolve_existing_root(args.mri_root, "MRI")
        pet_root  = resolve_existing_root(args.pet_root, "PET")
        print(f"[Info] ADNI phase: {adni_phase}")
        print(f"[Info] Match CSV: {match_csv}")
        print(f"[Info] MRI root: {mri_root}")
        print(f"[Info] PET root: {pet_root}")
        samples = load_adni_mri_pet_samples(
            match_csv=match_csv,
            mri_root=mri_root,
            pet_root=pet_root,
            adni_phase=adni_phase,
        )

    print(f"[Info] Samples after task filter: {len(samples)}  dist={samples['label_name'].value_counts().to_dict()}")

    samples = maybe_subsample(
        samples=samples,
        max_samples=args.max_samples,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
    )

    # ── Split ─────────────────────────────────────────────────────────────────
    subject_split_csv = str(getattr(args, "subject_split_csv", "")).strip()
    if subject_split_csv:
        print(f"[Info] Using shared subject split: {subject_split_csv}")
        train_df, val_df, test_df = _apply_subject_split(samples, subject_split_csv)
    else:
        train_df, val_df, test_df = split_by_subject(
            samples=samples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        train_df, val_df, test_df = rebalance_split_class_counts_by_subject(
            samples=samples,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    print(f"[Info] Train rows: {len(train_df)} | subjects: {train_df['Subject_ID'].nunique()}")
    print(f"[Info] Val rows: {len(val_df)} | subjects: {val_df['Subject_ID'].nunique()}")
    print(f"[Info] Test rows: {len(test_df)} | subjects: {test_df['Subject_ID'].nunique()}")
    _print_split_label_distribution("Train", train_df)
    _print_split_label_distribution("Val", val_df)
    _print_split_label_distribution("Test", test_df)

    all_samples_path = os.path.join(args.output_dir, "resolved_adni_mri_pet_samples.csv")
    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_assignments_path = os.path.join(args.output_dir, "adni_split_assignments.csv")
    train_subject_ids_path = os.path.join(args.output_dir, "train_subject_ids.txt")
    val_subject_ids_path = os.path.join(args.output_dir, "val_subject_ids.txt")
    test_subject_ids_path = os.path.join(args.output_dir, "test_subject_ids.txt")
    samples.to_csv(all_samples_path, index=False)
    split_df.to_csv(split_assignments_path, index=False)
    for split_frame, output_path in (
        (train_df, train_subject_ids_path),
        (val_df, val_subject_ids_path),
        (test_df, test_subject_ids_path),
    ):
        subject_ids = sorted(split_frame["Subject_ID"].dropna().astype(str).unique().tolist())
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("\n".join(subject_ids))
            if subject_ids:
                file.write("\n")

    train_ds = MRIPEPairDataset(train_df, target_shape=target_shape, augment=bool(args.augment))
    train_eval_ds = MRIPEPairDataset(train_df, target_shape=target_shape, augment=False)
    val_ds = MRIPEPairDataset(val_df, target_shape=target_shape, augment=False)
    test_ds = MRIPEPairDataset(test_df, target_shape=target_shape, augment=False)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = MRIPEOTClassifier(
        num_classes=NUM_CLASSES,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
        ot_epsilon=args.ot_epsilon,
        ot_iters=args.ot_iters,
        encoder_backbone=args.encoder_backbone,
        mri_brainmvp_ckpt=resolved_mri_brainmvp_ckpt,
        pet_brainmvp_ckpt=resolved_pet_brainmvp_ckpt,
        brainmvp_trainable=bool(args.brainmvp_trainable),
        brainmvp_input_shape=args.depth,
        classwise_ot=bool(args.classwise_ot),
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        lora_target_suffixes=tuple(
            s.strip() for s in str(args.lora_target).split(",") if s.strip()
        ),
    ).to(device)
    class_counts = train_df["label"].value_counts().reindex(range(NUM_CLASSES)).fillna(0).to_numpy()
    class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for class_idx, count in enumerate(class_counts):
        class_weights[class_idx] = 0.0 if count == 0 else len(train_df) / (NUM_CLASSES * count)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_macro_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    history_csv_path = os.path.join(args.output_dir, "training_history.csv")

    epoch_progress = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_progress:
        train_loss, train_cls_loss, train_ot_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            split_name="train",
            epoch_idx=epoch,
            total_epochs=args.epochs,
            ot_weight=args.ot_weight,
            require_cuda=(device.type == "cuda"),
        )
        val_loss, val_cls_loss, val_ot_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            split_name="val",
            epoch_idx=epoch,
            total_epochs=args.epochs,
            ot_weight=args.ot_weight,
            require_cuda=(device.type == "cuda"),
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_cls_loss": train_cls_loss,
            "val_cls_loss": val_cls_loss,
            "train_ot_loss": train_ot_loss,
            "val_ot_loss": val_ot_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "train_auc_roc_ovr_macro": train_metrics["auc_roc_ovr_macro"],
            "val_auc_roc_ovr_macro": val_metrics["auc_roc_ovr_macro"],
            "train_sensitivity_recall": train_metrics["sensitivity_macro"],
            "val_sensitivity_recall": val_metrics["sensitivity_macro"],
            "train_specificity": train_metrics["specificity_macro"],
            "val_specificity": val_metrics["specificity_macro"],
            "train_precision": train_metrics["precision_macro"],
            "val_precision": val_metrics["precision_macro"],
            "train_f1_score": train_metrics["f1_macro"],
            "val_f1_score": val_metrics["f1_macro"],
            "train_mcc": train_metrics["mcc"],
            "val_mcc": val_metrics["mcc"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_csv_path, index=False)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_cls={val_cls_loss:.4f} val_ot={val_ot_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"val_mcc={val_metrics['mcc']:.4f}"
        )
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            val_f1=f"{val_metrics['f1_macro']:.4f}",
        )

        if epoch % confusion_matrix_interval == 0:
            for split_name, split_metrics in (("train", train_metrics), ("val", val_metrics)):
                cm = split_metrics.get("confusion_matrix", [])
                if isinstance(cm, list) and cm:
                    cm_csv_path = os.path.join(
                        confusion_dir, f"{split_name}_confusion_matrix_epoch_{epoch:04d}.csv"
                    )
                    cm_png_path = os.path.join(
                        confusion_dir, f"{split_name}_confusion_matrix_epoch_{epoch:04d}.png"
                    )
                    save_confusion_matrix(cm_csv_path, cm, CLASS_NAMES)
                    save_confusion_matrix_png(
                        cm_png_path,
                        cm,
                        CLASS_NAMES,
                        title=f"{split_name.capitalize()} Confusion Matrix @ Epoch {epoch}",
                    )

        # Model selection metric: macro-F1 on validation split.
        if val_metrics["f1_macro"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    save_training_plots(history_df, args.output_dir)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_eval_loss, train_eval_cls, train_eval_ot, train_eval_metrics = run_epoch(
        model=model,
        loader=train_eval_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="train_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
        ot_weight=args.ot_weight,
        require_cuda=(device.type == "cuda"),
    )
    val_eval_loss, val_eval_cls, val_eval_ot, val_eval_metrics = run_epoch(
        model=model,
        loader=val_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="val_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
        ot_weight=args.ot_weight,
        require_cuda=(device.type == "cuda"),
    )
    test_loss, test_cls, test_ot, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="test_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
        ot_weight=args.ot_weight,
        require_cuda=(device.type == "cuda"),
    )

    final_eval = {
        "train": {
            "loss": train_eval_loss,
            "cls_loss": train_eval_cls,
            "ot_loss": train_eval_ot,
            "metrics": train_eval_metrics,
        },
        "val": {
            "loss": val_eval_loss,
            "cls_loss": val_eval_cls,
            "ot_loss": val_eval_ot,
            "metrics": val_eval_metrics,
        },
        "test": {
            "loss": test_loss,
            "cls_loss": test_cls,
            "ot_loss": test_ot,
            "metrics": test_metrics,
        },
    }

    for split_name, payload in final_eval.items():
        cm = payload["metrics"].get("confusion_matrix", [])
        if isinstance(cm, list) and cm:
            cm_csv_path = os.path.join(confusion_dir, f"{split_name}_confusion_matrix_final.csv")
            cm_png_path = os.path.join(confusion_dir, f"{split_name}_confusion_matrix_final.png")
            save_confusion_matrix(cm_csv_path, cm, CLASS_NAMES)
            save_confusion_matrix_png(
                cm_png_path,
                cm,
                CLASS_NAMES,
                title=f"{split_name.capitalize()} Confusion Matrix (Best Checkpoint)",
            )
            if split_name == "test":
                save_confusion_matrix(
                    os.path.join(args.output_dir, "test_confusion_matrix.csv"),
                    cm,
                    CLASS_NAMES,
                )
                save_confusion_matrix_png(
                    os.path.join(args.output_dir, "test_confusion_matrix.png"),
                    cm,
                    CLASS_NAMES,
                    title="Test Confusion Matrix",
                )

    split_rows = []
    for split_name, payload in final_eval.items():
        row = {
            "split": split_name,
            "loss": float(payload["loss"]),
            "cls_loss": float(payload["cls_loss"]),
            "ot_loss": float(payload["ot_loss"]),
        }
        row.update(selected_metric_view(payload["metrics"]))
        split_rows.append(row)
    split_report_df = pd.DataFrame(split_rows)
    split_report_path = os.path.join(args.output_dir, "split_metrics_report.csv")
    split_report_df.to_csv(split_report_path, index=False)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "task": args.task,
        "adni_phase": adni_phase,
        "match_csv": locals().get("match_csv", ""),
        "mri_root": locals().get("mri_root", ""),
        "pet_root": locals().get("pet_root", ""),
        "target_shape": list(target_shape),
        "feature_dim": int(args.feature_dim),
        "dropout": float(args.dropout),
        "encoder_backbone": str(args.encoder_backbone),
        "mri_brainmvp_ckpt": resolved_mri_brainmvp_ckpt,
        "pet_brainmvp_ckpt": resolved_pet_brainmvp_ckpt,
        "brainmvp_trainable": bool(args.brainmvp_trainable),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_target": str(args.lora_target),
        "classwise_ot": bool(args.classwise_ot),
        "ot_weight": float(args.ot_weight),
        "ot_epsilon": float(args.ot_epsilon),
        "ot_iters": int(args.ot_iters),
        "max_samples": int(args.max_samples),
        "samples_per_class": int(args.samples_per_class),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_subjects": int(train_df["Subject_ID"].nunique()),
        "val_subjects": int(val_df["Subject_ID"].nunique()),
        "test_subjects": int(test_df["Subject_ID"].nunique()),
        "class_names": CLASS_NAMES,
        "class_counts_train": {
            class_name: int((train_df["label"] == class_idx).sum())
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
        "best_val_macro_f1": float(best_val_macro_f1),
        "train_eval_loss": float(train_eval_loss),
        "val_eval_loss": float(val_eval_loss),
        "test_loss": float(test_loss),
        "train_core_metrics": selected_metric_view(train_eval_metrics),
        "val_core_metrics": selected_metric_view(val_eval_metrics),
        "test_core_metrics": selected_metric_view(test_metrics),
        "train_confusion_matrix": train_eval_metrics.get("confusion_matrix", []),
        "val_confusion_matrix": val_eval_metrics.get("confusion_matrix", []),
        "test_confusion_matrix": test_metrics.get("confusion_matrix", []),
        "resolved_samples_csv": all_samples_path,
        "split_assignments_csv": split_assignments_path,
        "train_subject_ids_txt": train_subject_ids_path,
        "val_subject_ids_txt": val_subject_ids_path,
        "test_subject_ids_txt": test_subject_ids_path,
        "split_metrics_csv": split_report_path,
        "best_model_path": best_model_path,
    }
    save_json(os.path.join(args.output_dir, "run_summary.json"), summary)

    print("\n[Done] Training complete.")
    print(f"[Done] Best model: {best_model_path}")
    print(f"[Done] Split metrics CSV: {split_report_path}")
    core = selected_metric_view(test_metrics)
    print(
        "[Done] Test metrics: "
        f"Accuracy={core['accuracy']:.4f}, "
        f"Balanced Accuracy={core['balanced_accuracy']:.4f}, "
        f"AUC-ROC={core['auc_roc']:.4f}, "
        f"Sensitivity/Recall={core['sensitivity_recall']:.4f}, "
        f"Specificity={core['specificity']:.4f}, "
        f"Precision={core['precision']:.4f}, "
        f"F1-score={core['f1_score']:.4f}, "
        f"MCC={core['mcc']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.task != "all":
        run_single_task(args)
        return

    task_sequence = ["binary_ad_cn", "ad_mci_cn"]
    base_output_dir = args.output_dir
    for task_name in task_sequence:
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task_name
        task_args.output_dir = os.path.join(base_output_dir, task_name)
        print(
            f"\n[Run] Starting task `{task_name}` "
            f"with output dir `{task_args.output_dir}`"
        )
        run_single_task(task_args)

    print("\n[Done] Completed both tasks in sequence: AD/CN then AD/MCI/CN.")


if __name__ == "__main__":
    main()

"""Train a 3D ResNet classifier for ADNI PET."""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models.video.resnet import BasicBlock, BasicStem, Bottleneck, Conv3DSimple, VideoResNet


DEFAULT_ADNI_PHASE = "ADNI3"
ADNI3_BETA_AMYLOID_MATCH_CSV_DEFAULT = (
    "data_preprocessing/adni3/paired_mri_t1_pet_uniform6mm_by_pet_target/"
    "adni3_mri_t1__pet_uniform6mm__pet_target_Beta_amyloid_PET__matched_subject_pairs__filesystem_matched_only.csv"
)
MATCH_CSV_DEFAULT = ADNI3_BETA_AMYLOID_MATCH_CSV_DEFAULT
ADNI1_MATCH_CSV_DEFAULT = MATCH_CSV_DEFAULT  # Backward-compat alias.
PET_ROOT_DEFAULT = "datasets/ADNI3/PET_preprocessed/ADNI"
PET_ROOT_FALLBACK = "datasets/ADNI3/PET_preprocessed_hdbet/ADNI"
PHASE_TO_CSV_PHASE_NAME = {"ADNI1": "ADNI 1", "ADNI2": "ADNI 2", "ADNI3": "ADNI 3", "ADNI4": "ADNI 4"}
SUBJECT_PATTERN = re.compile(r"^\d{3}_S_\d{4}$")
DATE_PREFIX_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
TASK_TO_CLASS_TO_LABEL = {
    "binary_ad_cn": {"CN": 0, "AD": 1},
    "three_class": {"CN": 0, "MCI": 1, "AD": 2},
    "ad_mci_cn": {"CN": 0, "MCI": 1, "AD": 2},
}
CLASS_TO_LABEL = TASK_TO_CLASS_TO_LABEL["binary_ad_cn"].copy()
NUM_CLASSES = len(CLASS_TO_LABEL)
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}
CLASS_NAMES = [LABEL_TO_CLASS[idx] for idx in range(NUM_CLASSES)]


def configure_label_space(task: str) -> None:
    """Configure global class mappings based on selected classification task."""
    global CLASS_TO_LABEL, NUM_CLASSES, LABEL_TO_CLASS, CLASS_NAMES
    if task not in TASK_TO_CLASS_TO_LABEL:
        raise ValueError(f"Unknown task: {task}")
    CLASS_TO_LABEL = TASK_TO_CLASS_TO_LABEL[task].copy()
    NUM_CLASSES = len(CLASS_TO_LABEL)
    LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}
    CLASS_NAMES = [LABEL_TO_CLASS[idx] for idx in range(NUM_CLASSES)]


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ADNI PET 3D ResNet (binary AD/CN or AD/MCI/CN)."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary_ad_cn", "ad_mci_cn", "all"],
        default="binary_ad_cn",
        help="Classification task: binary AD/CN, AD/MCI/CN, or all (run both).",
    )
    parser.add_argument(
        "--adni-phase",
        type=str,
        choices=["ADNI1", "ADNI2", "ADNI3", "ADNI4"],
        default=DEFAULT_ADNI_PHASE,
        help="ADNI phase used to auto-resolve default CSV/PET paths.",
    )
    parser.add_argument("--match-csv", type=str, default=MATCH_CSV_DEFAULT)
    parser.add_argument("--pet-root", "--mri-root", dest="pet_root", type=str, default=PET_ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=str, default="results/adni_pet_resnet3d")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet34", "resnet50", "resnet101"],
        default="resnet101",
        help="3D ResNet backbone depth.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for quick smoke runs. 0 means use all samples.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help=(
            "Optional balanced cap per class (without replacement). "
            "For example, 100 means 100 samples for each class. "
            "0 disables this option."
        ),
    )
    parser.add_argument(
        "--date-fallback-days",
        type=int,
        default=45,
        help="When exact CSV PET date is missing in files, allow nearest date within N days.",
    )
    parser.add_argument(
        "--confusion-matrix-interval",
        type=int,
        default=0,
        help="Save train/val confusion matrix PNG/CSV every N epochs (0 disables periodic save).",
    )
    parser.add_argument(
        "--subject-split-csv",
        type=str,
        default="",
        help=(
            "Optional shared subject-level split CSV with columns Subject_ID,split "
            "(split in {train,val,test}). If missing, this run creates it."
        ),
    )
    parser.add_argument(
        "--samples-csv",
        type=str,
        default="",
        help=(
            "Pre-resolved samples manifest CSV with columns Subject_ID, Research_Group, pet_path "
            "(and optionally PET_Date, PET_Description). When provided, bypasses --match-csv and "
            "--pet-root resolution entirely. Label mapping is applied based on --task."
        ),
    )
    return parser.parse_args()


def default_match_csv_for_phase(phase: str) -> str:
    if str(phase).upper() == "ADNI3":
        paired_csv = Path(ADNI3_BETA_AMYLOID_MATCH_CSV_DEFAULT).resolve()
        if paired_csv.is_file():
            return str(paired_csv)
    phase_csv = Path(f"datasets/ADNI/matches_from_phase_csvs/{phase}_MRIT1_PET_matches_baselinevisit.csv")
    if phase_csv.is_file():
        return str(phase_csv)
    return f"datasets/ADNI/matches_from_all_pairs/{phase}_MRIT1_PET_matches.csv"


def default_preprocessed_pet_root_for_phase(phase: str) -> str:
    return f"datasets/{phase}/PET_preprocessed/ADNI"


def fallback_preprocessed_pet_root_for_phase(phase: str) -> str:
    return f"datasets/{phase}/PET_preprocessed_hdbet/ADNI"


def default_raw_pet_root_for_phase(phase: str) -> str:
    return f"datasets/{phase}/PET/ADNI"


def resolve_pet_root(requested_root: str, phase: str) -> str:
    """Resolve PET root path with ADNI1 fallback for this workspace layout."""
    if os.path.isdir(requested_root):
        return requested_root
    fallbacks = [fallback_preprocessed_pet_root_for_phase(phase), PET_ROOT_FALLBACK]
    for fallback in fallbacks:
        if os.path.isdir(fallback):
            print(
                f"[Info] Requested PET root not found: {requested_root}\n"
                f"[Info] Falling back to: {fallback}"
            )
            return fallback
    raise FileNotFoundError(
        f"PET root not found: {requested_root} and no phase-specific fallback was found."
    )


def normalize_desc_tokens(description: str) -> List[str]:
    """Extract robust lowercase keyword tokens from PET description text."""
    if not isinstance(description, str):
        return []
    left = description.split("<-")[0].lower()
    left = left.replace(";", " ").replace("/", " ")
    raw_tokens = re.findall(r"[a-z0-9\-]+", left)
    return [tok for tok in raw_tokens if len(tok) >= 3]


def parse_csv_date_to_key(date_value: object) -> Optional[str]:
    """Convert CSV PET date to YYYY-MM-DD key used for path indexing."""
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d")


def extract_subject_and_date_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Extract subject ID and date key from nested ADNI PET path."""
    subject_id = None
    date_key = None
    for part in path.parts:
        if SUBJECT_PATTERN.match(part):
            subject_id = part
        if DATE_PREFIX_PATTERN.match(part):
            date_key = part[:10]
    return subject_id, date_key


def build_pet_index(pet_root: str) -> Tuple[Dict[Tuple[str, str], List[Path]], Dict[str, List[Tuple[str, Path]]]]:
    """Build indices for fast subject/date -> PET file lookup."""
    by_subject_date: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    by_subject: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

    root = Path(pet_root)
    nii_files = sorted(list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz")))
    if not nii_files:
        raise RuntimeError(f"No NIfTI files found under {pet_root}")

    for file_path in nii_files:
        subject_id, date_key = extract_subject_and_date_from_path(file_path)
        if subject_id is None or date_key is None:
            continue
        by_subject_date[(subject_id, date_key)].append(file_path)
        by_subject[subject_id].append((date_key, file_path))

    return by_subject_date, by_subject


def choose_best_candidate(candidates: List[Path], pet_description: str) -> Path:
    """Choose one PET file among candidates using description-token overlap."""
    if len(candidates) == 1:
        return candidates[0]

    tokens = normalize_desc_tokens(pet_description)
    scored = []
    for path in candidates:
        path_text = str(path).lower()
        score = sum(token in path_text for token in tokens)
        scored.append((score, str(path), path))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def find_nearest_date_candidate(
    target_date_key: str,
    subject_entries: List[Tuple[str, Path]],
    max_days: int,
) -> List[Path]:
    """Return all files from the nearest available date for a subject within a day threshold."""
    target_dt = pd.to_datetime(target_date_key, errors="coerce")
    if pd.isna(target_dt):
        return []

    by_date: Dict[str, List[Path]] = defaultdict(list)
    for date_key, path in subject_entries:
        by_date[date_key].append(path)

    candidates = []
    for date_key in by_date:
        dt = pd.to_datetime(date_key, errors="coerce")
        if pd.isna(dt):
            continue
        delta = abs((dt - target_dt).days)
        candidates.append((delta, date_key))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])
    best_delta, best_date_key = candidates[0]
    if best_delta > max_days:
        return []
    return by_date[best_date_key]


def resolve_pet_path_for_row(
    row: pd.Series,
    by_subject_date: Dict[Tuple[str, str], List[Path]],
    by_subject: Dict[str, List[Tuple[str, Path]]],
    max_fallback_days: int,
) -> Tuple[Optional[Path], str]:
    """Resolve MRI file path for one CSV row; returns (path, resolution_mode)."""
    subject_id = str(row["Subject_ID"])
    date_key = parse_csv_date_to_key(row["PET_Date"])
    if date_key is None:
        return None, "no_csv_date"

    direct_candidates = by_subject_date.get((subject_id, date_key), [])
    if direct_candidates:
        return choose_best_candidate(direct_candidates, str(row["PET_Description"])), "exact_date"

    subject_entries = by_subject.get(subject_id, [])
    if subject_entries:
        nearest_candidates = find_nearest_date_candidate(
            target_date_key=date_key,
            subject_entries=subject_entries,
            max_days=max_fallback_days,
        )
        if nearest_candidates:
            return (
                choose_best_candidate(nearest_candidates, str(row["PET_Description"])),
                "nearest_date_fallback",
            )

    return None, "not_found"


def build_label(research_group: object) -> Optional[int]:
    """Task-dependent label mapping using the configured CLASS_TO_LABEL."""
    if pd.isna(research_group):
        return None
    rg = str(research_group).strip().upper()
    return CLASS_TO_LABEL.get(rg)


def map_series_dir_to_preprocessed_path(
    series_dir: str,
    preprocessed_pet_root: str,
    adni_phase: str,
) -> Path:
    """Map raw PET series dir (..../PET/ADNI/.../Ixxxx) to preprocessed NIfTI path."""
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

    # PET preprocessing writes outputs under:
    # <pet_root>/<subject>/<desc>/<date>/<series_id>/<series_id>_preproc.nii.gz
    # so we keep the full raw series relative path (including series_id directory).
    out_parent = preproc_root / rel
    return out_parent / f"{series_path.name}_preproc.nii.gz"


def load_adni_pet_samples(
    match_csv: str,
    pet_root: str,
    max_fallback_days: int,
    adni_phase: str,
) -> pd.DataFrame:
    """Load ADNI phase CSV and resolve each row to a PET file path and task label."""
    match_df = pd.read_csv(match_csv)

    paired_required_cols = {
        "Subject ID",
        "PET_Study Date",
        "PET_Description",
        "PET_Research Group",
        "PET_Series_Dir",
    }
    is_paired_schema = paired_required_cols.issubset(set(match_df.columns))
    if is_paired_schema:
        rows = []
        resolution_counts = defaultdict(int)
        unresolved_rows = 0
        unknown_label_rows = 0
        for _, row in match_df.iterrows():
            label = build_label(row["PET_Research Group"])
            if label is None:
                unknown_label_rows += 1
                continue

            raw_series_dir = str(row["PET_Series_Dir"]).strip()
            if not raw_series_dir:
                unresolved_rows += 1
                resolution_counts["missing_pet_series_dir"] += 1
                continue

            pet_path = map_series_dir_to_preprocessed_path(
                series_dir=raw_series_dir,
                preprocessed_pet_root=pet_root,
                adni_phase=adni_phase,
            )
            if not pet_path.is_file():
                unresolved_rows += 1
                resolution_counts["preprocessed_not_found"] += 1
                continue

            resolution_counts["paired_series_dir_to_preprocessed"] += 1
            rows.append(
                {
                    "Subject_ID": str(row["Subject ID"]),
                    "PET_Date": str(row["PET_Study Date"]),
                    "PET_Description": str(row["PET_Description"]),
                    "Research Group": str(row["PET_Research Group"]),
                    "label": int(label),
                    "label_name": LABEL_TO_CLASS[int(label)],
                    "pet_path": str(pet_path),
                    "resolution_mode": "paired_series_dir_to_preprocessed",
                }
            )

        samples = pd.DataFrame(rows)
        if samples.empty:
            raise RuntimeError("No PET samples could be resolved from paired-series CSV.")

        before_dedup = len(samples)
        samples = samples.sort_values(["Subject_ID", "PET_Date", "pet_path"]).drop_duplicates(
            subset=["pet_path"], keep="first"
        )

        print(f"[Info] Input match rows: {len(match_df)}")
        print(f"[Info] Resolved PET rows before dedup: {before_dedup}")
        print(f"[Info] PET rows after dedup by path: {len(samples)}")
        print(f"[Info] Unresolved rows: {unresolved_rows}")
        print(f"[Info] Unknown-label rows dropped: {unknown_label_rows}")
        print(f"[Info] Resolution mode counts: {dict(resolution_counts)}")
        print("[Info] Label distribution:")
        print(samples["label_name"].value_counts().to_string())
        return samples.reset_index(drop=True)

    if "Phase" in match_df.columns:
        phase_name = PHASE_TO_CSV_PHASE_NAME.get(adni_phase, adni_phase)
        match_df = match_df[match_df["Phase"].astype(str).str.strip() == phase_name].copy()

    required_cols = {"Subject_ID", "PET_Date", "PET_Description", "Research Group"}
    missing_cols = required_cols - set(match_df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in {match_csv}: {sorted(missing_cols)}")

    by_subject_date, by_subject = build_pet_index(pet_root)

    rows = []
    resolution_counts = defaultdict(int)
    unresolved_rows = 0
    unknown_label_rows = 0

    for _, row in match_df.iterrows():
        label = build_label(row["Research Group"])
        if label is None:
            unknown_label_rows += 1
            continue

        pet_path, mode = resolve_pet_path_for_row(
            row=row,
            by_subject_date=by_subject_date,
            by_subject=by_subject,
            max_fallback_days=max_fallback_days,
        )
        resolution_counts[mode] += 1
        if pet_path is None:
            unresolved_rows += 1
            continue

        rows.append(
            {
                "Subject_ID": str(row["Subject_ID"]),
                "PET_Date": str(row["PET_Date"]),
                "PET_Description": str(row["PET_Description"]),
                "Research Group": str(row["Research Group"]),
                "label": int(label),
                "label_name": LABEL_TO_CLASS[int(label)],
                "pet_path": str(pet_path),
                "resolution_mode": mode,
            }
        )

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError("No PET samples could be resolved from the match CSV.")

    # PET-only training should not duplicate the exact same volume file.
    before_dedup = len(samples)
    samples = samples.sort_values(["Subject_ID", "PET_Date", "pet_path"]).drop_duplicates(
        subset=["pet_path"], keep="first"
    )

    print(f"[Info] Input match rows: {len(match_df)}")
    print(f"[Info] Resolved PET rows before dedup: {before_dedup}")
    print(f"[Info] PET rows after dedup by path: {len(samples)}")
    print(f"[Info] Unresolved rows: {unresolved_rows}")
    print(f"[Info] Unknown-label rows dropped: {unknown_label_rows}")
    print(f"[Info] Resolution mode counts: {dict(resolution_counts)}")
    print("[Info] Label distribution:")
    print(samples["label_name"].value_counts().to_string())

    return samples.reset_index(drop=True)



class PETVolumeDataset(Dataset):
    """Dataset that loads 3D NIfTI PET volumes and returns tensors."""

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

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: (1, D, H, W)
        tensor = tensor.unsqueeze(0)  # (N=1, C=1, D, H, W)
        tensor = F.interpolate(
            tensor,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return tensor.squeeze(0)

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        # Lightweight augmentations suitable for 3D PET.
        if torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[3])  # flip width
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(tensor) * 0.02
            tensor = tensor + noise
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        nii_path = row["pet_path"]
        label = int(row["label"])

        volume = nib.load(nii_path).get_fdata(dtype=np.float32)
        if volume.ndim == 4:
            volume = volume[..., 0]
        volume = self._normalize(volume)

        tensor = torch.from_numpy(volume).float().unsqueeze(0)
        tensor = self._resize(tensor)
        if self.augment:
            tensor = self._augment(tensor)

        return tensor, torch.tensor(label, dtype=torch.long)


def split_by_subject(
    samples: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by subject to avoid leakage across train/val/test."""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    subject_df = samples.groupby("Subject_ID", as_index=False).agg({"label": "first"})

    trainval_subj, test_subj = train_test_split(
        subject_df["Subject_ID"],
        test_size=test_ratio,
        random_state=seed,
        stratify=subject_df["label"],
    )

    trainval_df = subject_df[subject_df["Subject_ID"].isin(trainval_subj)].copy()
    val_relative_ratio = val_ratio / (1.0 - test_ratio)

    train_subj, val_subj = train_test_split(
        trainval_df["Subject_ID"],
        test_size=val_relative_ratio,
        random_state=seed,
        stratify=trainval_df["label"],
    )

    train_df = samples[samples["Subject_ID"].isin(set(train_subj))].copy()
    val_df = samples[samples["Subject_ID"].isin(set(val_subj))].copy()
    test_df = samples[samples["Subject_ID"].isin(set(test_subj))].copy()

    return train_df, val_df, test_df


def _load_subject_split_df(path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(path)
    required_cols = {"Subject_ID", "split"}
    missing = required_cols - set(split_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in shared split CSV {path}: {sorted(missing)}")

    split_df = split_df[["Subject_ID", "split"]].copy()
    split_df["Subject_ID"] = split_df["Subject_ID"].astype(str).str.strip()
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()
    split_df = split_df[split_df["Subject_ID"] != ""].copy()

    valid_splits = {"train", "val", "test"}
    invalid = sorted(set(split_df["split"]) - valid_splits)
    if invalid:
        raise ValueError(f"Invalid split labels in {path}: {invalid}. Expected only {sorted(valid_splits)}")

    duplicated = split_df[split_df.duplicated(subset=["Subject_ID"], keep=False)].copy()
    if not duplicated.empty:
        conflict_counts = duplicated.groupby("Subject_ID")["split"].nunique()
        conflicting_subjects = sorted(conflict_counts[conflict_counts > 1].index.tolist())
        if conflicting_subjects:
            preview = ", ".join(conflicting_subjects[:10])
            raise ValueError(
                f"Conflicting split assignments for {len(conflicting_subjects)} subjects in {path}. "
                f"Examples: {preview}"
            )
        split_df = split_df.drop_duplicates(subset=["Subject_ID"], keep="first")

    return split_df.reset_index(drop=True)


def _build_subject_split_df(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        subject_ids = sorted(frame["Subject_ID"].dropna().astype(str).str.strip().unique().tolist())
        rows.extend({"Subject_ID": sid, "split": split_name} for sid in subject_ids)
    return pd.DataFrame(rows, columns=["Subject_ID", "split"])


def split_with_optional_subject_csv(
    samples: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    subject_split_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    split_csv = str(subject_split_csv).strip()
    if not split_csv:
        train_df, val_df, test_df = split_by_subject(
            samples=samples,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        return train_df, val_df, test_df, ""

    split_path = Path(split_csv).expanduser().resolve()
    if split_path.is_file():
        split_df = _load_subject_split_df(split_path)
        sample_subjects = set(samples["Subject_ID"].dropna().astype(str).str.strip().tolist())
        known_subjects = set(split_df["Subject_ID"].tolist())
        missing_subjects = sorted(sample_subjects - known_subjects)
        if missing_subjects:
            preview = ", ".join(missing_subjects[:10])
            raise ValueError(
                f"Shared subject split CSV is missing {len(missing_subjects)} subjects for this task. "
                f"Examples: {preview}"
            )

        split_df = split_df[split_df["Subject_ID"].isin(sample_subjects)].copy()
        split_map = split_df.set_index("Subject_ID")["split"].to_dict()
        subject_series = samples["Subject_ID"].astype(str).str.strip()
        split_series = subject_series.map(split_map)
        train_df = samples[split_series == "train"].copy()
        val_df = samples[split_series == "val"].copy()
        test_df = samples[split_series == "test"].copy()
        print(f"[Info] Loaded shared subject split CSV: {split_path}")
    else:
        train_df, val_df, test_df = split_by_subject(
            samples=samples,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split_df = _build_subject_split_df(train_df=train_df, val_df=val_df, test_df=test_df)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_path, index=False)
        print(f"[Info] Created shared subject split CSV: {split_path}")

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            "Shared split produced an empty train/val/test subset for this task. "
            "Adjust split ratios or regenerate the shared split."
        )

    return train_df, val_df, test_df, str(split_path)


_BACKBONE_CONFIGS = {
    "resnet34":  {"block": BasicBlock, "layers": [3, 4,  6,  3]},
    "resnet50":  {"block": Bottleneck, "layers": [3, 4,  6,  3]},
    "resnet101": {"block": Bottleneck, "layers": [3, 4, 23,  3]},
}


def create_model(num_classes: int, backbone: str = "resnet101") -> nn.Module:
    """Create a 3D ResNet (34/50/101) and adapt input/output layers for single-channel PET."""
    cfg = _BACKBONE_CONFIGS[backbone]
    model = VideoResNet(
        block=cfg["block"],
        conv_makers=[Conv3DSimple] * 4,
        layers=cfg["layers"],
        stem=BasicStem,
        num_classes=400,
    )

    # Change first conv from 3-channel RGB to single-channel PET input.
    old_conv = model.stem[0]
    new_conv = nn.Conv3d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.stem[0] = new_conv

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def multiclass_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute multiclass metrics with torchmetrics (macro + per-class + AUC + MCC)."""
    if len(y_true) == 0:
        metrics: Dict[str, object] = {
            "accuracy": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "specificity_macro": float("nan"),
            "f1_macro": float("nan"),
            "balanced_accuracy": float("nan"),
            "mcc": float("nan"),
            "auc_roc_ovr_macro": float("nan"),
            "auc_roc_ovr_weighted": float("nan"),
            "confusion_matrix": [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)],
        }
        for class_name in CLASS_NAMES:
            metrics[f"precision_{class_name}"] = float("nan")
            metrics[f"recall_{class_name}"] = float("nan")
            metrics[f"specificity_{class_name}"] = float("nan")
            metrics[f"f1_{class_name}"] = float("nan")
            metrics[f"support_{class_name}"] = 0.0
        return metrics

    y_true_tensor = torch.tensor(y_true, dtype=torch.long)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.long)
    y_prob_tensor = (
        torch.tensor(y_prob, dtype=torch.float32)
        if y_prob is not None and len(y_prob) == len(y_true)
        else None
    )

    acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro")
    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average="macro")
    recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average="macro")
    specificity_metric = MulticlassSpecificity(num_classes=NUM_CLASSES, average="macro")
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
    mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES)

    precision_per_class_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average=None)
    recall_per_class_metric = MulticlassRecall(num_classes=NUM_CLASSES, average=None)
    specificity_per_class_metric = MulticlassSpecificity(num_classes=NUM_CLASSES, average=None)
    f1_per_class_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average=None)

    accuracy = float(acc_metric(y_pred_tensor, y_true_tensor).item())
    macro_precision = float(precision_metric(y_pred_tensor, y_true_tensor).item())
    macro_recall = float(recall_metric(y_pred_tensor, y_true_tensor).item())
    macro_specificity = float(specificity_metric(y_pred_tensor, y_true_tensor).item())
    macro_f1 = float(f1_metric(y_pred_tensor, y_true_tensor).item())

    try:
        mcc_value = float(mcc_metric(y_pred_tensor, y_true_tensor).item())
    except Exception:
        mcc_value = float("nan")

    precision_per_class = precision_per_class_metric(y_pred_tensor, y_true_tensor).tolist()
    recall_per_class = recall_per_class_metric(y_pred_tensor, y_true_tensor).tolist()
    specificity_per_class = specificity_per_class_metric(y_pred_tensor, y_true_tensor).tolist()
    f1_per_class = f1_per_class_metric(y_pred_tensor, y_true_tensor).tolist()
    supports = torch.bincount(y_true_tensor, minlength=NUM_CLASSES).float().tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    metrics = {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "specificity_macro": macro_specificity,
        "f1_macro": macro_f1,
        # Balanced accuracy = macro recall in multiclass settings.
        "balanced_accuracy": macro_recall,
        "mcc": mcc_value,
        "confusion_matrix": cm.tolist(),
    }

    if NUM_CLASSES == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel().tolist()
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
        metrics["tp"] = float(tp)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_precision = float(precision_per_class[class_idx])
        class_recall = float(recall_per_class[class_idx])
        class_specificity = float(specificity_per_class[class_idx])
        class_f1 = float(f1_per_class[class_idx])
        class_support = float(supports[class_idx])

        metrics[f"precision_{class_name}"] = class_precision
        metrics[f"recall_{class_name}"] = class_recall
        metrics[f"specificity_{class_name}"] = class_specificity
        metrics[f"f1_{class_name}"] = class_f1
        metrics[f"support_{class_name}"] = class_support

    if y_prob_tensor is not None:
        try:
            auc_macro_metric = MulticlassAUROC(num_classes=NUM_CLASSES, average="macro")
            metrics["auc_roc_ovr_macro"] = float(auc_macro_metric(y_prob_tensor, y_true_tensor).item())
        except Exception:
            metrics["auc_roc_ovr_macro"] = float("nan")

        try:
            auc_weighted_metric = MulticlassAUROC(num_classes=NUM_CLASSES, average="weighted")
            metrics["auc_roc_ovr_weighted"] = float(
                auc_weighted_metric(y_prob_tensor, y_true_tensor).item()
            )
        except Exception:
            metrics["auc_roc_ovr_weighted"] = float("nan")
    else:
        metrics["auc_roc_ovr_macro"] = float("nan")
        metrics["auc_roc_ovr_weighted"] = float("nan")

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
) -> Tuple[float, Dict[str, object]]:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []

    progress = tqdm(
        loader,
        desc=f"{split_name} {epoch_idx}/{total_epochs}",
        leave=False,
    )
    for volumes, labels in progress:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(volumes)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * volumes.size(0)
        total_count += int(volumes.size(0))
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())
        all_prob.append(probs.detach().cpu().numpy())
        progress.set_postfix(batch_loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_count, 1)
    prob_array = np.concatenate(all_prob, axis=0) if all_prob else None
    metrics = multiclass_metrics(all_true, all_pred, y_prob=prob_array)
    return avg_loss, metrics


def maybe_subsample(
    samples: pd.DataFrame,
    max_samples: int,
    seed: int,
    samples_per_class: int = 0,
) -> pd.DataFrame:
    """Optionally subsample for quick tests while preserving class balance roughly."""
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

    # Keep class ratio approximately by sampling each class proportionally.
    for label, group_df in samples.groupby("label"):
        frac = len(group_df) / len(samples)
        take = max(1, int(round(max_samples * frac)))
        take = min(take, len(group_df))
        sampled_idx = rng.choice(group_df.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(samples.loc[sampled_idx])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    return sampled.reset_index(drop=True)


def save_json(path: str, payload: dict) -> None:
    """Save JSON with UTF-8 and indentation."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_confusion_matrix(path: str, matrix: List[List[int]], class_names: List[str]) -> None:
    """Save confusion matrix as CSV with class labels as rows/columns."""
    cm = np.asarray(matrix, dtype=np.int64)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(path, index=True)


def build_confusion_matrix_figure(matrix: List[List[int]], class_names: List[str], title: str):
    """Build a matplotlib confusion matrix figure."""
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
    return fig


def save_confusion_matrix_png(
    path: str,
    matrix: List[List[int]],
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix heatmap as a PNG file using matplotlib."""
    import matplotlib.pyplot as plt

    fig = build_confusion_matrix_figure(matrix, class_names, title=title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_training_plots(history_df: pd.DataFrame, output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import os
    
    metrics_to_plot = [
        ("loss", "loss"),
        ("accuracy", "accuracy"),
        ("balanced_accuracy", "balanced_accuracy"),
        ("auc_roc_ovr_macro", "auc_roc"),
        ("sensitivity_recall", "sensitivity"),
        ("specificity", "specificity"),
        ("precision", "precision"),
        ("f1_score", "f1_score"),
        ("mcc", "mcc"),
    ]
    epochs = history_df["epoch"]
    
    for raw_metric, clean_name in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        train_col = f"train_{raw_metric}"
        val_col = f"val_{raw_metric}"
        test_col = f"test_{raw_metric}"
        
        if train_col in history_df.columns:
            ax.plot(epochs, history_df[train_col], label="Train", linewidth=2)
        if val_col in history_df.columns:
            ax.plot(epochs, history_df[val_col], label="Val", linewidth=2)
            
        if test_col in history_df.columns:
            # Drop nans to only plot points where test was evaluated
            test_df = history_df.dropna(subset=[test_col])
            if not test_df.empty:
                ax.plot(test_df["epoch"], test_df[test_col], marker='o', linestyle='--', label="Test", linewidth=2)
                
        ax.set_xlabel("Epoch")
        ax.set_ylabel(clean_name.capitalize())
        ax.set_title(f"Training Progress: {clean_name.capitalize()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"plot_{clean_name}.png"), bbox_inches="tight")
        plt.close(fig)



def save_split_metric_plot(split_report_df: pd.DataFrame, output_path: str) -> None:
    """Save bar plot for train/val/test final metric values."""
    if split_report_df.empty:
        return

    import matplotlib.pyplot as plt

    metrics_to_plot = [
        "accuracy",
        "balanced_accuracy",
        "auc_roc",
        "sensitivity_recall",
        "specificity",
        "precision",
        "f1_score",
        "mcc",
    ]

    splits = split_report_df["split"].tolist()
    n_metrics = len(metrics_to_plot)
    x = np.arange(len(splits))
    width = 0.8 / max(n_metrics, 1)

    all_values: List[float] = []
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    for idx, metric_name in enumerate(metrics_to_plot):
        if metric_name not in split_report_df.columns:
            continue
        values = split_report_df[metric_name].astype(float).to_numpy()
        all_values.extend(values.tolist())
        offset = (idx - (n_metrics - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)
        pad = max(0.05, (ymax - ymin) * 0.15)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("Metric value")
    ax.set_title("Final Split Metrics (Train/Val/Test)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classifier_layer: nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return penultimate-layer embeddings and ground-truth labels for all samples in loader."""
    embeddings: List[np.ndarray] = []
    labels: List[int] = []

    def _hook(module: nn.Module, inp: Tuple, out: torch.Tensor) -> None:  # noqa: ARG001
        embeddings.append(inp[0].detach().cpu().numpy())

    handle = classifier_layer.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            volumes = batch[0].to(device)
            batch_labels = batch[1].numpy()
            model(volumes)
            labels.append(batch_labels)
    handle.remove()

    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def save_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: str,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """Fit t-SNE on embeddings and save a scatter plot coloured by class."""
    import matplotlib.pyplot as plt

    n_samples = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(1.0, n_samples - 1))
    coords = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        max_iter=1000,
    ).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    palette = plt.cm.get_cmap("tab10", len(class_names))
    for idx, name in enumerate(class_names):
        mask = labels == idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=name,
            color=palette(idx),
            s=18,
            alpha=0.75,
            linewidths=0,
        )
    ax.set_title("t-SNE — Test Set Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def selected_metric_view(metrics: Dict[str, object]) -> Dict[str, float]:
    """Return the core metric set requested for reporting."""
    return {
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "auc_roc": float(metrics.get("auc_roc_ovr_macro", float("nan"))),
        "sensitivity_recall": float(metrics.get("recall_macro", float("nan"))),
        "specificity": float(metrics.get("specificity_macro", float("nan"))),
        "precision": float(metrics.get("precision_macro", float("nan"))),
        "f1_score": float(metrics.get("f1_macro", float("nan"))),
        "mcc": float(metrics.get("mcc", float("nan"))),
    }


def run_single_task(args: argparse.Namespace) -> None:
    adni_phase = str(args.adni_phase).strip().upper()
    if args.match_csv == MATCH_CSV_DEFAULT:
        args.match_csv = default_match_csv_for_phase(adni_phase)
    if args.pet_root == PET_ROOT_DEFAULT:
        args.pet_root = default_preprocessed_pet_root_for_phase(adni_phase)
    if args.confusion_matrix_interval < 0:
        raise ValueError("--confusion-matrix-interval must be >= 0.")

    configure_label_space(args.task)
    set_seed(args.seed)

    match_csv = args.match_csv
    os.makedirs(args.output_dir, exist_ok=True)
    confusion_dir = os.path.join(args.output_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_shape = (args.depth, args.height, args.width)

    print(f"[Info] Device: {device}")
    print(f"[Info] Task: {args.task}")
    print(f"[Info] ADNI phase: {adni_phase}")
    print(f"[Info] Classes: {CLASS_NAMES}")
    print(f"[Info] Target volume shape: {target_shape}")

    samples_csv_path = str(getattr(args, "samples_csv", "")).strip()
    if samples_csv_path:
        print(f"[Info] Loading samples from --samples-csv: {samples_csv_path}")
        raw_manifest = pd.read_csv(samples_csv_path)
        required_manifest_cols = {"Subject_ID", "Research_Group", "pet_path"}
        missing_cols = required_manifest_cols - set(raw_manifest.columns)
        if missing_cols:
            raise KeyError(f"--samples-csv is missing required columns: {sorted(missing_cols)}")
        rows = []
        skipped = 0
        for _, row in raw_manifest.iterrows():
            label = build_label(row["Research_Group"])
            if label is None:
                skipped += 1
                continue
            rows.append({
                "Subject_ID": str(row["Subject_ID"]).strip(),
                "PET_Date": str(row.get("PET_Date", "")),
                "PET_Description": str(row.get("PET_Description", "")),
                "Research Group": str(row["Research_Group"]).strip(),
                "label": int(label),
                "label_name": LABEL_TO_CLASS[int(label)],
                "pet_path": str(row["pet_path"]).strip(),
                "resolution_mode": "samples_csv",
            })
        if skipped:
            print(f"[Info] Dropped {skipped} rows not valid for task={args.task}")
        samples = pd.DataFrame(rows).drop_duplicates(subset=["pet_path"], keep="first").reset_index(drop=True)
        if samples.empty:
            raise RuntimeError(f"No samples remain after label filtering for task={args.task}.")
        print(f"[Info] Loaded {len(samples)} samples ({samples['Subject_ID'].nunique()} subjects)")
        print("[Info] Label distribution:")
        print(samples["label_name"].value_counts().to_string())
        pet_root = "n/a (--samples-csv)"
    else:
        pet_root = resolve_pet_root(args.pet_root, adni_phase)
        print(f"[Info] Match CSV: {match_csv}")
        print(f"[Info] PET root: {pet_root}")
        samples = load_adni_pet_samples(
            match_csv=match_csv,
            pet_root=pet_root,
            max_fallback_days=args.date_fallback_days,
            adni_phase=adni_phase,
        )
    samples = maybe_subsample(
        samples=samples,
        max_samples=args.max_samples,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
    )

    train_df, val_df, test_df, shared_subject_split_csv = split_with_optional_subject_csv(
        samples=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        subject_split_csv=args.subject_split_csv,
    )

    print(f"[Info] Train rows: {len(train_df)} | subjects: {train_df['Subject_ID'].nunique()}")
    print("[Info] Train Label Distribution:")
    print(train_df["label_name"].value_counts().to_string())
    
    print(f"[Info] Val rows: {len(val_df)} | subjects: {val_df['Subject_ID'].nunique()}")
    print("[Info] Val Label Distribution:")
    print(val_df["label_name"].value_counts().to_string())
    
    print(f"[Info] Test rows: {len(test_df)} | subjects: {test_df['Subject_ID'].nunique()}")
    print("[Info] Test Label Distribution:")
    print(test_df["label_name"].value_counts().to_string())

    # Persist resolved samples and split assignments for reproducibility.
    all_samples_path = os.path.join(args.output_dir, "resolved_adni_pet_samples.csv")
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

    train_ds = PETVolumeDataset(train_df, target_shape=target_shape, augment=True)
    train_eval_ds = PETVolumeDataset(train_df, target_shape=target_shape, augment=False)
    val_ds = PETVolumeDataset(val_df, target_shape=target_shape, augment=False)
    test_ds = PETVolumeDataset(test_df, target_shape=target_shape, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(num_classes=NUM_CLASSES, backbone=args.backbone).to(device)

    class_counts = train_df["label"].value_counts().reindex(range(NUM_CLASSES)).fillna(0).to_numpy()
    class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for class_idx, count in enumerate(class_counts):
        class_weights[class_idx] = 0.0 if count == 0 else len(train_df) / (NUM_CLASSES * count)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_macro_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    epoch_progress = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_progress:
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            split_name="train",
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            split_name="val",
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        
        # Periodic test eval
        test_loss, test_metrics = float('nan'), None
        if epoch % 5 == 0 or epoch == args.epochs:
            test_loss, test_metrics = run_epoch(
                model=model,
                loader=test_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                split_name="test",
                epoch_idx=epoch,
                total_epochs=args.epochs,
            )
            
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"] if test_metrics is not None else float('nan'),
            
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"] if test_metrics is not None else float('nan'),
            
            "train_auc_roc_ovr_macro": train_metrics["auc_roc_ovr_macro"],
            "val_auc_roc_ovr_macro": val_metrics["auc_roc_ovr_macro"],
            "test_auc_roc_ovr_macro": test_metrics["auc_roc_ovr_macro"] if test_metrics is not None else float('nan'),
            
            "train_sensitivity_recall": train_metrics["recall_macro"],
            "val_sensitivity_recall": val_metrics["recall_macro"],
            "test_sensitivity_recall": test_metrics["recall_macro"] if test_metrics is not None else float('nan'),
            
            "train_specificity": train_metrics["specificity_macro"],
            "val_specificity": val_metrics["specificity_macro"],
            "test_specificity": test_metrics["specificity_macro"] if test_metrics is not None else float('nan'),
            
            "train_precision": train_metrics["precision_macro"],
            "val_precision": val_metrics["precision_macro"],
            "test_precision": test_metrics["precision_macro"] if test_metrics is not None else float('nan'),
            
            "train_f1_score": train_metrics["f1_macro"],
            "val_f1_score": val_metrics["f1_macro"],
            "test_f1_score": test_metrics["f1_macro"] if test_metrics is not None else float('nan'),
            
            "train_mcc": train_metrics["mcc"],
            "val_mcc": val_metrics["mcc"],
            "test_mcc": test_metrics["mcc"] if test_metrics is not None else float('nan'),
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"val_auc={val_metrics['auc_roc_ovr_macro']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"val_mcc={val_metrics['mcc']:.4f}"
        )
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            train_acc=f"{train_metrics['accuracy']:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            val_mcc=f"{val_metrics['mcc']:.4f}",
        )

        should_save_periodic_cm = (
            args.confusion_matrix_interval > 0 and epoch % args.confusion_matrix_interval == 0
        )
        if should_save_periodic_cm:
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

        if val_metrics["f1_macro"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)
    save_training_plots(history_df, args.output_dir)

    # Final split evaluation using best validation checkpoint.
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_eval_loss, train_eval_metrics = run_epoch(
        model=model,
        loader=train_eval_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="train_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
    )
    val_eval_loss, val_eval_metrics = run_epoch(
        model=model,
        loader=val_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="val_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
    )
    test_loss, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="test_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
    )

    final_eval = {
        "train": {"loss": train_eval_loss, "metrics": train_eval_metrics},
        "val": {"loss": val_eval_loss, "metrics": val_eval_metrics},
        "test": {"loss": test_loss, "metrics": test_metrics},
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
                # Keep backward-compatible artifact locations.
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
        }
        row.update(selected_metric_view(payload["metrics"]))
        split_rows.append(row)
    split_report_df = pd.DataFrame(split_rows)
    split_report_path = os.path.join(args.output_dir, "split_metrics_report.csv")
    split_report_df.to_csv(split_report_path, index=False)
    save_split_metric_plot(
        split_report_df=split_report_df,
        output_path=os.path.join(args.output_dir, "final_split_metrics.png"),
    )

    test_embeddings, test_embedding_labels = extract_embeddings(
        model=model,
        loader=test_loader,
        device=device,
        classifier_layer=model.fc,
    )
    save_tsne_plot(
        embeddings=test_embeddings,
        labels=test_embedding_labels,
        class_names=CLASS_NAMES,
        output_path=os.path.join(args.output_dir, "tsne_test.png"),
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "task": args.task,
        "adni_phase": adni_phase,
        "match_csv": match_csv,
        "pet_root": pet_root,
        "target_shape": list(target_shape),
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
        "shared_subject_split_csv": shared_subject_split_csv,
        "test_metrics": test_metrics,
        "best_model_path": best_model_path,
    }
    save_json(os.path.join(args.output_dir, "run_summary.json"), summary)

    print("\n[Done] Training complete.")
    print(f"[Done] Best model: {best_model_path}")
    print(f"[Done] Split metrics CSV: {split_report_path}")
    core = selected_metric_view(test_metrics)
    print(
        f"[Done] Test confusion matrix (rows=true, cols=pred): "
        f"{test_metrics.get('confusion_matrix', [])}"
    )
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

    # Requested order: AD/CN first, then AD/MCI/CN.
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

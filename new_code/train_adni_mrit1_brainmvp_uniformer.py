"""Train ADNI PET classifier with a BrainMVP UniFormer backbone.

Workflow:
1. Resolve ADNI rows to PET files using the existing project pipeline.
2. Build a UniFormer classifier head on top of BrainMVP's backbone.
3. Load BrainMVP pretrained weights (backbone only).
4. Train/evaluate and save metrics, plots, and confusion matrices.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm

import train_adni_mrit1_resnet3d as base_train


DEFAULT_PRETRAINED = "BrainMVP/BrainMVP_uniformer.pt"


def _load_uniformer_small():
    """Load `uniformer_small` directly from BrainMVP's downstream model file."""
    module_path = Path(__file__).resolve().parent / "BrainMVP" / "Downstream" / "model" / "uniformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"BrainMVP UniFormer module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("brainmvp_uniformer_module", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.uniformer_small


class BrainMVPUniFormerClassifier(nn.Module):
    """UniFormer backbone + global average pooling + linear classification head."""

    def __init__(self, num_classes: int, in_channels: int, input_shape: int, dropout: float = 0.2):
        super().__init__()
        uniformer_small = _load_uniformer_small()
        self.backbone = uniformer_small(img_size=input_shape, in_chans=in_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base dataset tensors are B,C,D,H,W. BrainMVP expects B,C,H,W,D before its internal permute.
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        _, _, _, _, x4 = self.backbone(x)
        pooled = F.adaptive_avg_pool3d(x4, output_size=1).flatten(1)
        return self.classifier(self.dropout(pooled))


class LoRALinear(nn.Module):
    """LoRA wrapper around a frozen `nn.Linear` layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0.")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


def _inject_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    """Replace linear layers inside `module` with LoRA adapters recursively."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += _inject_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return replaced


def configure_peft(model: BrainMVPUniFormerClassifier, args: argparse.Namespace) -> None:
    """Configure full fine-tuning or LoRA PEFT for the backbone."""
    if args.peft_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        print("[Info] PEFT mode: full (all parameters trainable).")
        return

    if args.peft_mode == "lora":
        # Freeze backbone first, then add LoRA adapters to its Linear layers.
        for param in model.backbone.parameters():
            param.requires_grad = False
        replaced = _inject_lora(
            model.backbone,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        for param in model.classifier.parameters():
            param.requires_grad = True
        print(
            "[Info] PEFT mode: lora "
            f"(rank={args.lora_rank}, alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}, replaced_linear_layers={replaced})."
        )
        return

    raise ValueError(f"Unknown --peft-mode: {args.peft_mode}")


def parse_args() -> argparse.Namespace:
    """Parse only the core arguments needed for training/evaluation."""
    parser = argparse.ArgumentParser(
        description="Train ADNI MRI T1 classifier with BrainMVP UniFormer pretrained backbone."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary_ad_cn", "ad_mci_cn", "all"],
        default="binary_ad_cn",
    )
    parser.add_argument(
        "--adni-phase",
        type=str,
        choices=["ADNI1", "ADNI2", "ADNI3", "ADNI4"],
        default=base_train.DEFAULT_ADNI_PHASE,
    )
    parser.add_argument("--match-csv", type=str, default=base_train.MATCH_CSV_DEFAULT)
    parser.add_argument("--mri-t1-root", dest="mri_t1_root", type=str, default=base_train.MRI_T1_ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=str, default="results/adni_mri_brainmvp_uniformer")
    parser.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument("--peft-mode", type=str, choices=["full", "lora"], default="full")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
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
    parser.add_argument("--date-fallback-days", type=int, default=45)
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
            "Pre-resolved samples manifest CSV with columns Subject_ID, Research_Group, mri_t1_path "
            "(and optionally PET_Date, PET_Description). When provided, bypasses --match-csv and "
            "--mri_t1-root resolution entirely. Label mapping is applied based on --task."
        ),
    )
    parser.add_argument(
        "--confusion-matrix-interval",
        type=int,
        default=0,
        help="Save train/val confusion matrix PNG/CSV every N epochs (0 disables periodic save).",
    )
    return parser.parse_args()


def _unwrap_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved as raw state_dict or wrapped dictionaries."""
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            maybe = ckpt_obj.get(key)  # type: ignore[arg-type]
            if isinstance(maybe, dict):
                return maybe
        if ckpt_obj and all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj  # type: ignore[return-value]
    raise RuntimeError("Unsupported checkpoint format. Expected dict or dict with `state_dict`.")


def _normalize_backbone_key(key: str) -> Optional[str]:
    """Normalize common prefixes so pretrained keys map to `model.backbone` keys."""
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

    # Drop non-backbone tensors from pretraining checkpoints.
    if k.startswith("decoder.") or k.startswith("rep_template"):
        return None
    return k


def _match_or_adapt_weight(key: str, src: torch.Tensor, dst: torch.Tensor) -> Optional[torch.Tensor]:
    """Match weight by shape; adapt first input conv if channel counts differ."""
    if src.shape == dst.shape:
        return src

    # BrainMVP pretraining often used 4 modalities; ADNI MRI-T1 here uses 1 channel.
    if key == "patch_embed1.proj.weight" and src.ndim == 5 and dst.ndim == 5:
        if src.shape[1] > 1 and dst.shape[1] == 1:
            return src.mean(dim=1, keepdim=True)
        if src.shape[1] == 1 and dst.shape[1] > 1:
            return src.repeat(1, dst.shape[1], 1, 1, 1) / float(dst.shape[1])
    return None


def load_pretrained_backbone(model: BrainMVPUniFormerClassifier, pretrained_path: str) -> Tuple[int, int, int]:
    """Load only compatible backbone tensors from pretrained checkpoint."""
    ckpt = torch.load(pretrained_path, map_location="cpu")
    src_state = _unwrap_state_dict(ckpt)
    dst_state = model.backbone.state_dict()

    loadable: Dict[str, torch.Tensor] = {}
    skipped = 0
    for raw_key, tensor in src_state.items():
        key = _normalize_backbone_key(raw_key)
        if key is None or key not in dst_state:
            skipped += 1
            continue
        adapted = _match_or_adapt_weight(key=key, src=tensor, dst=dst_state[key])
        if adapted is None or adapted.shape != dst_state[key].shape:
            skipped += 1
            continue
        loadable[key] = adapted

    missing, unexpected = model.backbone.load_state_dict(loadable, strict=False)
    loaded = len(loadable)
    unresolved = len(missing) + len(unexpected)
    return loaded, skipped, unresolved


def build_model(args: argparse.Namespace, num_classes: int) -> BrainMVPUniFormerClassifier:
    """Build classifier and optionally load pretrained backbone weights."""
    model = BrainMVPUniFormerClassifier(
        num_classes=num_classes,
        in_channels=1,
        input_shape=args.depth,
        dropout=args.dropout,
    )

    ckpt_path = args.pretrained.strip()
    if not ckpt_path:
        print("[Info] No --pretrained path provided; training from scratch.")
        return model
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    loaded, skipped, unresolved = load_pretrained_backbone(model, ckpt_path)
    print(
        f"[Info] Pretrained backbone loaded from {ckpt_path} | "
        f"loaded={loaded}, skipped={skipped}, unresolved_after_load={unresolved}"
    )
    return model


def _build_subject_split_df(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        subject_ids = sorted(frame["Subject_ID"].dropna().astype(str).str.strip().unique().tolist())
        rows.extend({"Subject_ID": sid, "split": split_name} for sid in subject_ids)
    return pd.DataFrame(rows, columns=["Subject_ID", "split"])


def _build_row_split_df(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        if "mri_t1_path" not in frame.columns:
            raise KeyError("mri_t1_path column is required to build row-level split mapping.")
        part = frame[["mri_t1_path"]].copy()
        if "Subject_ID" in frame.columns:
            part["Subject_ID"] = frame["Subject_ID"].astype(str).str.strip()
        part["split"] = split_name
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["mri_t1_path"] = out["mri_t1_path"].astype(str).str.strip()
    out = out[out["mri_t1_path"] != ""].copy()
    out = out.drop_duplicates(subset=["mri_t1_path"], keep="first")
    return out.reset_index(drop=True)


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


def _load_row_split_df(path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(path)
    required_cols = {"mri_t1_path", "split"}
    missing = required_cols - set(split_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in shared row split CSV {path}: {sorted(missing)}")

    keep_cols = ["mri_t1_path", "split"]
    if "Subject_ID" in split_df.columns:
        keep_cols.append("Subject_ID")
    split_df = split_df[keep_cols].copy()
    split_df["mri_t1_path"] = split_df["mri_t1_path"].astype(str).str.strip()
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()
    split_df = split_df[split_df["mri_t1_path"] != ""].copy()

    valid_splits = {"train", "val", "test"}
    invalid = sorted(set(split_df["split"]) - valid_splits)
    if invalid:
        raise ValueError(f"Invalid split labels in {path}: {invalid}. Expected only {sorted(valid_splits)}")

    duplicated = split_df[split_df.duplicated(subset=["mri_t1_path"], keep=False)].copy()
    if not duplicated.empty:
        conflict_counts = duplicated.groupby("mri_t1_path")["split"].nunique()
        conflicting_paths = sorted(conflict_counts[conflict_counts > 1].index.tolist())
        if conflicting_paths:
            preview = ", ".join(conflicting_paths[:5])
            raise ValueError(
                f"Conflicting split assignments for {len(conflicting_paths)} mri_t1_path entries in {path}. "
                f"Examples: {preview}"
            )
        split_df = split_df.drop_duplicates(subset=["mri_t1_path"], keep="first")

    return split_df.reset_index(drop=True)


def _load_shared_split_df(path: Path) -> Tuple[str, pd.DataFrame]:
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    if {"mri_t1_path", "split"}.issubset(cols):
        return "row", _load_row_split_df(path)
    if {"Subject_ID", "split"}.issubset(cols):
        return "subject", _load_subject_split_df(path)
    raise KeyError(
        f"Unsupported shared split CSV schema in {path}. "
        "Expected columns [mri_t1_path, split] or [Subject_ID, split]."
    )


def _split_rows_exact_per_class(
    samples: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    train_parts = []
    val_parts = []
    test_parts = []
    for label in sorted(samples["label"].dropna().astype(int).unique().tolist()):
        group_df = samples[samples["label"] == label].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group_df)
        if n == 0:
            continue

        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if (n_test + n_val) >= n:
            n_test = min(n_test, max(0, n - 1))
            n_val = min(n_val, max(0, n - n_test - 1))
        n_train = n - n_val - n_test

        train_parts.append(group_df.iloc[:n_train].copy())
        val_parts.append(group_df.iloc[n_train : n_train + n_val].copy())
        test_parts.append(group_df.iloc[n_train + n_val :].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def _split_with_shared_subject_csv(
    samples: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    subject_split_csv: str,
    use_exact_row_split: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    split_csv = str(subject_split_csv).strip()
    if not split_csv:
        if use_exact_row_split:
            train_df, val_df, test_df = _split_rows_exact_per_class(
                samples=samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
        else:
            train_df, val_df, test_df = base_train.split_by_subject(
                samples=samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
        return train_df, val_df, test_df, ""

    split_path = Path(split_csv).expanduser().resolve()

    if split_path.is_file():
        split_kind, split_df = _load_shared_split_df(split_path)
        if split_kind == "row":
            if "mri_t1_path" not in samples.columns:
                raise KeyError("mri_t1_path column not found in samples for row-level shared split application.")
            sample_paths = set(samples["mri_t1_path"].astype(str).str.strip().tolist())
            known_paths = set(split_df["mri_t1_path"].tolist())
            missing_paths = sorted(sample_paths - known_paths)
            if missing_paths:
                preview = ", ".join(missing_paths[:5])
                raise ValueError(
                    f"Shared row split CSV is missing {len(missing_paths)} rows for this task. "
                    f"Examples: {preview}. Generate the shared split from the superset task first."
                )
            split_df = split_df[split_df["mri_t1_path"].isin(sample_paths)].copy()
            split_map = split_df.set_index("mri_t1_path")["split"].to_dict()
            path_series = samples["mri_t1_path"].astype(str).str.strip()
            split_series = path_series.map(split_map)
            train_df = samples[split_series == "train"].copy()
            val_df = samples[split_series == "val"].copy()
            test_df = samples[split_series == "test"].copy()
            print(f"[Info] Loaded shared row split CSV: {split_path}")
        else:
            sample_subjects = set(samples["Subject_ID"].dropna().astype(str).str.strip().tolist())
            known_subjects = set(split_df["Subject_ID"].tolist())
            missing_subjects = sorted(sample_subjects - known_subjects)
            if missing_subjects:
                preview = ", ".join(missing_subjects[:10])
                raise ValueError(
                    f"Shared subject split CSV is missing {len(missing_subjects)} subjects for this task. "
                    f"Examples: {preview}. Generate the shared split from the superset task first."
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
        if use_exact_row_split:
            train_df, val_df, test_df = _split_rows_exact_per_class(
                samples=samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            split_df = _build_row_split_df(train_df=train_df, val_df=val_df, test_df=test_df)
            split_label = "row"
        else:
            train_df, val_df, test_df = base_train.split_by_subject(
                samples=samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            split_df = _build_subject_split_df(train_df=train_df, val_df=val_df, test_df=test_df)
            split_label = "subject"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_path, index=False)
        print(f"[Info] Created shared {split_label} split CSV: {split_path}")

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            "Shared split produced an empty train/val/test subset for this task. "
            "Adjust split ratios or regenerate the shared split."
        )

    return train_df, val_df, test_df, str(split_path)


def _filter_samples_to_existing_shared_split(samples: pd.DataFrame, subject_split_csv: str) -> pd.DataFrame:
    split_csv = str(subject_split_csv).strip()
    if not split_csv:
        return samples

    split_path = Path(split_csv).expanduser().resolve()
    if not split_path.is_file():
        return samples

    split_kind, split_df = _load_shared_split_df(split_path)
    if split_kind == "row":
        if "mri_t1_path" not in samples.columns:
            return samples
        allowed_paths = set(split_df["mri_t1_path"].tolist())
        path_series = samples["mri_t1_path"].astype(str).str.strip()
        filtered = samples[path_series.isin(allowed_paths)].copy()
        dropped = len(samples) - len(filtered)
        if dropped > 0:
            print(
                f"[Info] Filtered out {dropped} rows not present in shared split rows: "
                f"{split_path}"
            )
        return filtered

    allowed_subjects = set(split_df["Subject_ID"].tolist())
    subject_series = samples["Subject_ID"].astype(str).str.strip()
    filtered = samples[subject_series.isin(allowed_subjects)].copy()
    dropped = len(samples) - len(filtered)
    if dropped > 0:
        print(
            f"[Info] Filtered out {dropped} rows not present in shared split subjects: "
            f"{split_path}"
        )
    return filtered


def run_single_task(args: argparse.Namespace) -> None:
    """Run one task configuration end-to-end (train, eval, save artifacts)."""
    adni_phase = str(args.adni_phase).strip().upper()
    if args.match_csv == base_train.MATCH_CSV_DEFAULT:
        args.match_csv = base_train.default_match_csv_for_phase(adni_phase)
    if args.mri_t1_root == base_train.MRI_T1_ROOT_DEFAULT:
        args.mri_t1_root = base_train.default_preprocessed_mri_t1_root_for_phase(adni_phase)

    base_train.configure_label_space(args.task)
    base_train.set_seed(args.seed)

    match_csv = args.match_csv
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_shape = (args.depth, args.height, args.width)

    print(f"[Info] Device: {device}")
    print(f"[Info] Task: {args.task}")
    print(f"[Info] Classes: {base_train.CLASS_NAMES}")

    samples_csv_path = str(getattr(args, "samples_csv", "")).strip()
    if samples_csv_path:
        print(f"[Info] Loading samples from --samples-csv: {samples_csv_path}")
        raw_manifest = pd.read_csv(samples_csv_path)
        required_cols = {"Subject_ID", "Research_Group", "mri_t1_path"}
        missing_cols = required_cols - set(raw_manifest.columns)
        if missing_cols:
            raise KeyError(f"--samples-csv is missing required columns: {sorted(missing_cols)}")
        rows = []
        skipped = 0
        for _, row in raw_manifest.iterrows():
            label = base_train.build_label(row["Research_Group"])
            if label is None:
                skipped += 1
                continue
            rows.append({
                "Subject_ID":      str(row["Subject_ID"]).strip(),
                "PET_Date":        str(row.get("PET_Date", "")),
                "PET_Description": str(row.get("PET_Description", "")),
                "Research Group":  str(row["Research_Group"]).strip(),
                "label":           int(label),
                "label_name":      base_train.LABEL_TO_CLASS[int(label)],
                "mri_t1_path":        str(row["mri_t1_path"]).strip(),
                "resolution_mode": "samples_csv",
            })
        if skipped:
            print(f"[Info] Dropped {skipped} rows not valid for task={args.task}")
        samples = pd.DataFrame(rows).drop_duplicates(subset=["mri_t1_path"], keep="first").reset_index(drop=True)
        if samples.empty:
            raise RuntimeError(f"No samples remain after label filtering for task={args.task}.")
        print(f"[Info] Loaded {len(samples)} samples ({samples['Subject_ID'].nunique()} subjects)")
        print("[Info] Label distribution:")
        print(samples["label_name"].value_counts().to_string())
        mri_t1_root = "n/a (--samples-csv)"
    else:
        mri_t1_root = base_train.resolve_mri_t1_root(args.mri_t1_root, adni_phase)
        print(f"[Info] Match CSV: {match_csv}")
        print(f"[Info] PET root: {mri_t1_root}")
        samples = base_train.load_adni_mri_t1_samples(
            match_csv=match_csv,
            mri_t1_root=mri_t1_root,
            max_fallback_days=args.date_fallback_days,
            adni_phase=adni_phase,
        )
    samples = base_train.maybe_subsample(
        samples=samples,
        max_samples=args.max_samples,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
    )

    train_df, val_df, test_df, shared_subject_split_csv = base_train.split_with_optional_subject_csv(
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

    all_samples_path = os.path.join(args.output_dir, "resolved_adni_mri_t1_samples.csv")
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

    train_ds = base_train.MRIT1VolumeDataset(train_df, target_shape=target_shape, augment=True)
    train_eval_ds = base_train.MRIT1VolumeDataset(train_df, target_shape=target_shape, augment=False)
    val_ds = base_train.MRIT1VolumeDataset(val_df, target_shape=target_shape, augment=False)
    test_ds = base_train.MRIT1VolumeDataset(test_df, target_shape=target_shape, augment=False)

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

    model = build_model(args=args, num_classes=base_train.NUM_CLASSES)
    configure_peft(model=model, args=args)
    model = model.to(device)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Trainable parameters: {trainable_params:,}")
    # Inverse-frequency class weights help under label imbalance.
    class_counts = train_df["label"].value_counts().reindex(range(base_train.NUM_CLASSES)).fillna(0).to_numpy()
    class_weights = np.zeros(base_train.NUM_CLASSES, dtype=np.float32)
    for class_idx, count in enumerate(class_counts):
        class_weights[class_idx] = 0.0 if count == 0 else len(train_df) / (base_train.NUM_CLASSES * count)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_param_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_macro_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_loss, train_metrics = base_train.run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            split_name="train",
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_metrics = base_train.run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            split_name="val",
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        
        test_loss, test_metrics = float('nan'), None
        if epoch % 5 == 0 or epoch == args.epochs:
            test_loss, test_metrics = base_train.run_epoch(
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

        history.append(
            {
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
        )

        if val_metrics["f1_macro"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_model_path)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f} "
            f"val_auc={val_metrics['auc_roc_ovr_macro']:.4f} val_mcc={val_metrics['mcc']:.4f}"
        )

    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    base_train.save_training_plots(history_df, args.output_dir)

    # Final evaluation uses the best validation checkpoint.
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_eval_loss, train_eval_metrics = base_train.run_epoch(
        model=model,
        loader=train_eval_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="train_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
    )
    val_eval_loss, val_eval_metrics = base_train.run_epoch(
        model=model,
        loader=val_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        split_name="val_eval",
        epoch_idx=args.epochs,
        total_epochs=args.epochs,
    )
    test_loss, test_metrics = base_train.run_epoch(
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

    confusion_dir = os.path.join(args.output_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    for split_name, payload in final_eval.items():
        cm = payload["metrics"].get("confusion_matrix", [])
        if isinstance(cm, list) and cm:
            base_train.save_confusion_matrix(
                os.path.join(confusion_dir, f"{split_name}_confusion_matrix_final.csv"),
                cm,
                base_train.CLASS_NAMES,
            )
            base_train.save_confusion_matrix_png(
                os.path.join(confusion_dir, f"{split_name}_confusion_matrix_final.png"),
                cm,
                base_train.CLASS_NAMES,
                title=f"{split_name.capitalize()} Confusion Matrix (Best Checkpoint)",
            )

    split_rows = []
    for split_name, payload in final_eval.items():
        row = {"split": split_name, "loss": float(payload["loss"])}
        row.update(base_train.selected_metric_view(payload["metrics"]))
        split_rows.append(row)
    split_report_df = pd.DataFrame(split_rows)
    split_report_path = os.path.join(args.output_dir, "split_metrics_report.csv")
    split_report_df.to_csv(split_report_path, index=False)
    base_train.save_split_metric_plot(
        split_report_df=split_report_df,
        output_path=os.path.join(args.output_dir, "final_split_metrics.png"),
    )

    test_embeddings, test_embedding_labels = base_train.extract_embeddings(
        model=model,
        loader=test_loader,
        device=device,
        classifier_layer=model.classifier,
    )
    base_train.save_tsne_plot(
        embeddings=test_embeddings,
        labels=test_embedding_labels,
        class_names=base_train.CLASS_NAMES,
        output_path=os.path.join(args.output_dir, "tsne_test.png"),
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "task": args.task,
        "adni_phase": adni_phase,
        "match_csv": match_csv,
        "mri_t1_root": args.mri_t1_root,
        "pretrained": args.pretrained,
        "peft_mode": args.peft_mode,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "target_shape": list(target_shape),
        "max_samples": int(args.max_samples),
        "samples_per_class": int(args.samples_per_class),
        "best_val_macro_f1": float(best_val_macro_f1),
        "train_eval_loss": float(train_eval_loss),
        "val_eval_loss": float(val_eval_loss),
        "test_loss": float(test_loss),
        "train_core_metrics": base_train.selected_metric_view(train_eval_metrics),
        "val_core_metrics": base_train.selected_metric_view(val_eval_metrics),
        "test_core_metrics": base_train.selected_metric_view(test_metrics),
        "resolved_samples_csv": all_samples_path,
        "split_assignments_csv": split_assignments_path,
        "train_subject_ids_txt": train_subject_ids_path,
        "val_subject_ids_txt": val_subject_ids_path,
        "test_subject_ids_txt": test_subject_ids_path,
        "best_model_path": best_model_path,
        "history_csv": history_path,
        "split_metrics_csv": split_report_path,
        "shared_subject_split_csv": shared_subject_split_csv,
    }
    base_train.save_json(os.path.join(args.output_dir, "run_summary.json"), summary)

    print("\n[Done] Training complete.")
    print(f"[Done] Best model: {best_model_path}")
    print(f"[Done] Split metrics CSV: {split_report_path}")
    test_core = base_train.selected_metric_view(test_metrics)
    print(
        "[Done] Test metrics: "
        f"Accuracy={test_core['accuracy']:.4f}, "
        f"Balanced Accuracy={test_core['balanced_accuracy']:.4f}, "
        f"AUC-ROC={test_core['auc_roc']:.4f}, "
        f"Sensitivity/Recall={test_core['sensitivity_recall']:.4f}, "
        f"Specificity={test_core['specificity']:.4f}, "
        f"Precision={test_core['precision']:.4f}, "
        f"F1-score={test_core['f1_score']:.4f}, "
        f"MCC={test_core['mcc']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.task != "all":
        run_single_task(args)
        return

    # "all" keeps the original behavior: run AD/CN then AD/MCI/CN.
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

"""Run multimodal MRI+PET OT experiments on ADNI3+ADNI4 Abeta-PET and Tau-PET.

Data sources (mirrors run_adni_pet_classification.py exactly):
  AD:      ADNI3 + ADNI4 (both modalities)
  MCI, CN: ADNI4 only

Both phases are combined into a single manifest per modality with a shared
train/val/test split, then one OT experiment is run per modality × task.

Output structure:
  <base-output-dir>/<modality>/multimodal_ot/<task>/<run-tag>/

Usage:
    python run_adni_ot_experiments.py
    python run_adni_ot_experiments.py --modality abeta_pet --task binary_ad_cn
    python run_adni_ot_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from train_adni_mrit1_pet_ot import (
    map_mri_series_dir_to_preprocessed_path,
    map_pet_series_dir_to_preprocessed_path,
    run_single_task,
)

# ===========================================================================
# ── CONFIGURATION ────────────────────────────────────────────────────────────
# ===========================================================================

# ── Input CSVs (paired MRI/PET manifests, one per ADNI phase × modality) ───
ADNI3_ABETA_CSV = (
    "/mnt/nvme_home/OT/data_preprocessing/adni3/paired_mri_t1_pet_uniform6mm_by_pet_target/"
    "adni3_mri_t1__pet_uniform6mm__pet_target_Beta_amyloid_PET__matched_subject_pairs__filesystem_matched_only.csv"
)
ADNI4_ABETA_CSV = (
    "/mnt/nvme_home/OT/data_preprocessing/adni4/paired_mri_t1_pet_uniform6mm_by_pet_target/"
    "adni4_mri_t1__pet_uniform6mm__pet_target_Beta_amyloid_PET__matched_subject_pairs__filesystem_matched_only.csv"
)
ADNI3_TAU_CSV = (
    "/mnt/nvme_home/OT/data_preprocessing/adni3/paired_mri_t1_pet_uniform6mm_by_pet_target/"
    "adni3_mri_t1__pet_uniform6mm__pet_target_Tau_PET__matched_subject_pairs__filesystem_matched_only.csv"
)
ADNI4_TAU_CSV = (
    "/mnt/nvme_home/OT/data_preprocessing/adni4/paired_mri_t1_pet_uniform6mm_by_pet_target/"
    "adni4_mri_t1__pet_uniform6mm__pet_target_Tau_PET__matched_subject_pairs__filesystem_matched_only.csv"
)

# ── Dataset roots ─────────────────────────────────────────────────────────────
ADNI3_MRI_ROOT = "/mnt/nvme_home/OT/datasets/ADNI3/MRIT1_freesurfer_preprocessed/ADNI"
ADNI3_PET_ROOT = "/mnt/nvme_home/OT/datasets/ADNI3/PET_preprocessed/ADNI"
ADNI4_MRI_ROOT = "/mnt/nvme_home/OT/datasets/ADNI4/MRIT1_freesurfer_preprocessed/ADNI"
ADNI4_PET_ROOT = "/mnt/nvme_home/OT/datasets/ADNI4/PET_preprocessed/ADNI"

# ── BrainMVP checkpoint templates ────────────────────────────────────────────
# {task} is replaced at runtime; * glob is used to find the latest run-tag subdir.
DEFAULT_MRI_BRAINMVP_CKPT = (
    "/mnt/nvme_home/OT/results/adni_freesurfer_mri_classification/"
    "mri_t1/brainmvp_uniformer/{task}/*/best_model.pt"
)
# {modality} and {task} are both replaced at runtime.
DEFAULT_PET_BRAINMVP_CKPT = (
    "/mnt/nvme_home/OT/results/adni_pet_classification/"
    "{modality}/brainmvp_uniformer/{task}/*/best_model.pt"
)

# ── Output directory ──────────────────────────────────────────────────────────
DEFAULT_BASE_OUTPUT = "/mnt/nvme_home/OT/results/adni_ot_experiments"

# ===========================================================================
# ── INTERNAL CONSTANTS ────────────────────────────────────────────────────────
# ===========================================================================

# 3-class universe for shared test split (mirrors run_adni_pet_classification.py)
CLASS_3 = {"CN": 0, "MCI": 1, "AD": 2}

TASK_CLASS_MAPS: Dict[str, Dict[str, int]] = {
    "ad_mci_cn":    {"CN": 0, "MCI": 1, "AD": 2},
    "binary_ad_cn": {"CN": 0, "AD": 1},
}
TASKS = list(TASK_CLASS_MAPS.keys())

MODALITY_CONFIGS: Dict[str, Dict] = {
    "abeta_pet": {
        "display": "Abeta-PET",
        "sources": [
            {
                "phase":        "ADNI3",
                "csv":          ADNI3_ABETA_CSV,
                "mri_root":     ADNI3_MRI_ROOT,
                "pet_root":     ADNI3_PET_ROOT,
                "adni_phase":   "ADNI3",
                "class_filter": {"AD"},   # ADNI3: AD only
            },
            {
                "phase":        "ADNI4",
                "csv":          ADNI4_ABETA_CSV,
                "mri_root":     ADNI4_MRI_ROOT,
                "pet_root":     ADNI4_PET_ROOT,
                "adni_phase":   "ADNI4",
                "class_filter": None,     # ADNI4: CN / MCI / AD
            },
        ],
    },
    "tau_pet": {
        "display": "Tau-PET",
        "sources": [
            {
                "phase":        "ADNI3",
                "csv":          ADNI3_TAU_CSV,
                "mri_root":     ADNI3_MRI_ROOT,
                "pet_root":     ADNI3_PET_ROOT,
                "adni_phase":   "ADNI3",
                "class_filter": {"AD"},
            },
            {
                "phase":        "ADNI4",
                "csv":          ADNI4_TAU_CSV,
                "mri_root":     ADNI4_MRI_ROOT,
                "pet_root":     ADNI4_PET_ROOT,
                "adni_phase":   "ADNI4",
                "class_filter": None,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Manifest building (paired MRI + PET paths)
# ---------------------------------------------------------------------------

def _load_phase_mri_pet_samples(
    csv_path: str,
    mri_root: str,
    pet_root: str,
    adni_phase: str,
    class_filter: Optional[Set[str]],
) -> pd.DataFrame:
    """Load one ADNI phase match CSV and resolve both MRI and PET preprocessed paths."""
    df = pd.read_csv(csv_path)
    # Normalize column names (spaces → underscores)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    rows = []
    missing = 0
    for _, row in df.iterrows():
        rg = str(row.get("MRI_Research_Group", row.get("Research_Group", ""))).strip().upper()
        if not rg or rg == "NAN":
            continue
        if class_filter is not None and rg not in class_filter:
            continue

        mri_series_dir = str(row.get("MRI_Series_Dir", "")).strip()
        pet_series_dir = str(row.get("PET_Series_Dir", "")).strip()
        if not mri_series_dir or not pet_series_dir:
            missing += 1
            continue

        mri_path = map_mri_series_dir_to_preprocessed_path(mri_series_dir, mri_root, adni_phase)
        pet_path = map_pet_series_dir_to_preprocessed_path(pet_series_dir, pet_root, adni_phase)

        if not mri_path.is_file() or not pet_path.is_file():
            missing += 1
            continue

        rows.append({
            "Subject_ID":     str(row.get("Subject_ID", row.get("Subject_id", ""))).strip(),
            "Research_Group": rg,
            "phase":          adni_phase,
            "mri_path":       str(mri_path),
            "pet_path":       str(pet_path),
        })

    result = pd.DataFrame(rows)
    if missing:
        print(f"    [Warn] {missing}/{len(df)} rows had no resolved MRI+PET file pair")
    return result


def build_modality_samples(modality: str) -> pd.DataFrame:
    """Combine ADNI3 (AD only) + ADNI4 (all classes) for the given modality."""
    cfg = MODALITY_CONFIGS[modality]
    parts = []
    for src in cfg["sources"]:
        cf = {c.upper() for c in src["class_filter"]} if src["class_filter"] else None
        cf_label = f"{sorted(cf)}" if cf else "all"
        print(f"  [{src['phase']}] classes={cf_label}  csv={Path(src['csv']).name}")
        part = _load_phase_mri_pet_samples(
            csv_path=src["csv"],
            mri_root=src["mri_root"],
            pet_root=src["pet_root"],
            adni_phase=src["adni_phase"],
            class_filter=cf,
        )
        dist = part["Research_Group"].value_counts().to_dict() if not part.empty else {}
        print(f"    Resolved {len(part)} samples, {part['Subject_ID'].nunique()} subjects  dist={dist}")
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined[combined["Subject_ID"] != ""].copy()
    combined = combined.drop_duplicates(subset=["mri_path", "pet_path"], keep="first").reset_index(drop=True)
    dist = combined["Research_Group"].value_counts().to_dict()
    print(f"  Combined: {len(combined)} samples, {combined['Subject_ID'].nunique()} subjects  dist={dist}")
    return combined


# ---------------------------------------------------------------------------
# Shared split building (mirrors run_adni_pet_classification.py)
# ---------------------------------------------------------------------------

def _subject_label_table(samples: pd.DataFrame) -> pd.DataFrame:
    valid = samples[samples["Research_Group"].isin(CLASS_3)].copy()
    valid["label"] = valid["Research_Group"].map(CLASS_3)
    return valid.groupby("Subject_ID", as_index=False).agg(
        {"label": "first", "Research_Group": "first"}
    ).reset_index(drop=True)


def build_shared_test_subjects(samples: pd.DataFrame, test_ratio: float, seed: int) -> Set[str]:
    subj_df = _subject_label_table(samples)
    _, test_subj = train_test_split(
        subj_df["Subject_ID"],
        test_size=test_ratio,
        random_state=seed,
        stratify=subj_df["label"],
    )
    test_set = set(test_subj.tolist())
    dist = {rg: 0 for rg in CLASS_3}
    for sid in test_set:
        rg = subj_df.loc[subj_df["Subject_ID"] == sid, "Research_Group"].iloc[0]
        dist[rg] = dist.get(rg, 0) + 1
    print(f"  Shared test subjects: {len(test_set)}  dist={dist}")
    return test_set


def build_shared_subject_split(
    samples: pd.DataFrame,
    test_subjects: Set[str],
    val_ratio: float,
    seed: int,
    split_root: Path,
) -> Path:
    subj_df = _subject_label_table(samples)
    test_df     = subj_df[subj_df["Subject_ID"].isin(test_subjects)].copy()
    trainval_df = subj_df[~subj_df["Subject_ID"].isin(test_subjects)].copy()

    if trainval_df.empty:
        raise RuntimeError("No train/val subjects left after test split.")

    try:
        train_ids, val_ids = train_test_split(
            trainval_df["Subject_ID"],
            test_size=val_ratio,
            random_state=seed,
            stratify=trainval_df["label"],
        )
    except ValueError as exc:
        print(f"  [Warn] Stratified val split fallback ({exc}). Using unstratified split.")
        train_ids, val_ids = train_test_split(
            trainval_df["Subject_ID"],
            test_size=val_ratio,
            random_state=seed,
        )

    split_df = pd.concat(
        [
            trainval_df[trainval_df["Subject_ID"].isin(set(train_ids))][["Subject_ID"]].assign(split="train"),
            trainval_df[trainval_df["Subject_ID"].isin(set(val_ids))][["Subject_ID"]].assign(split="val"),
            test_df[["Subject_ID"]].assign(split="test"),
        ],
        ignore_index=True,
    )

    split_root.mkdir(parents=True, exist_ok=True)
    split_csv = split_root / "shared_subject__split.csv"
    split_df.to_csv(split_csv, index=False)

    n_train = (split_df["split"] == "train").sum()
    n_val   = (split_df["split"] == "val").sum()
    n_test  = (split_df["split"] == "test").sum()
    print(f"  Split saved: train={n_train} val={n_val} test={n_test}  → {split_csv}")
    return split_csv


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_ckpt(template: str, task: str, modality: str) -> str:
    """Expand {task}/{modality} placeholders and resolve glob to the newest file."""
    path = template.format(task=task, modality=modality)
    if "*" not in path:
        return path
    matches = sorted(Path("/").glob(path.lstrip("/")), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found matching: {path}\n"
            "Pass explicit --mri-brainmvp-ckpt / --pet-brainmvp-ckpt to override."
        )
    resolved = str(matches[-1])
    print(f"  [Ckpt] {Path(template).parts[-1]} ({task}) → {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multimodal MRI+PET OT experiments on ADNI3+ADNI4 "
            "Abeta-PET and Tau-PET."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--modality", choices=["abeta_pet", "tau_pet", "all"], default="all",
    )
    parser.add_argument(
        "--task", choices=["binary_ad_cn", "ad_mci_cn", "all"], default="all",
    )
    parser.add_argument("--base-output-dir", type=str, default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--run-tag",         type=str, default="")

    # ── BrainMVP checkpoints ─────────────────────────────────────────────────
    parser.add_argument(
        "--encoder-backbone", choices=["small3d", "brainmvp"], default="brainmvp",
    )
    parser.add_argument("--mri-brainmvp-ckpt", type=str, default=DEFAULT_MRI_BRAINMVP_CKPT,
                        help="Supports {task} placeholder.")
    parser.add_argument("--pet-brainmvp-ckpt", type=str, default=DEFAULT_PET_BRAINMVP_CKPT,
                        help="Supports {task} placeholder.")
    parser.add_argument(
        "--brainmvp-trainable",
        type=lambda x: x.lower() not in ("0", "false", "no"),
        default=True,
        metavar="BOOL",
        help="Fine-tune BrainMVP backbones (default: True). Pass false/0/no to freeze.",
    )
    parser.add_argument("--lora-rank",  type=int,   default=8,
                        help="LoRA rank (default: 8). Set 0 to disable LoRA (full fine-tune).")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor (default: 16).")
    parser.add_argument("--lora-target", type=str, default="attn.qkv,attn.proj",
                        help="Comma-separated linear name suffixes to apply LoRA to.")

    # ── Data split ───────────────────────────────────────────────────────────
    parser.add_argument("--val-ratio",          type=float, default=0.15)
    parser.add_argument("--test-ratio",         type=float, default=0.15)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--max-samples",        type=int,   default=0)
    parser.add_argument("--samples-per-class",  type=int,   default=0)

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--num-workers",  type=int,   default=2)
    parser.add_argument("--depth",        type=int,   default=128)
    parser.add_argument("--height",       type=int,   default=128)
    parser.add_argument("--width",        type=int,   default=128)
    parser.add_argument("--feature-dim",  type=int,   default=256)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="cuda",
    )

    # ── OT hyperparameters ───────────────────────────────────────────────────
    parser.add_argument("--ot-weight",  type=float, default=0.1)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iters",   type=int,   default=40)
    parser.add_argument(
        "--classwise-ot",
        type=lambda x: x.lower() not in ("0", "false", "no"),
        default=True,
        metavar="BOOL",
        help="Compute OT per class (default: True). Pass false/0/no for batch-level OT.",
    )
    parser.add_argument("--confusion-matrix-interval", type=int, default=1)

    # ── Execution control ─────────────────────────────────────────────────────
    parser.add_argument("--augment",           action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Build manifests and splits but skip training.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token(value: object) -> str:
    return (
        str(value).strip().lower()
        .replace(".", "p").replace("-", "m").replace("+", "")
        .replace("/", "_").replace(" ", "")
    )


def _effective_run_tag(args: argparse.Namespace) -> str:
    if args.run_tag:
        return args.run_tag
    date_str = datetime.now().strftime("%Y%m%d")
    return (
        f"{date_str}_ep{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{_token(args.lr)}"
        f"_wd{_token(args.weight_decay)}"
        f"_fd{args.feature_dim}"
        f"_otw{_token(args.ot_weight)}"
        f"_ote{_token(args.ot_epsilon)}"
        f"_oti{args.ot_iters}"
        f"_seed{args.seed}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    base_output_dir = Path(args.base_output_dir).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = _effective_run_tag(args)

    modalities = list(MODALITY_CONFIGS.keys()) if args.modality == "all" else [args.modality]
    tasks      = TASKS if args.task == "all" else [args.task]

    print(f"\n{'='*70}")
    print(f"  Run tag     : {run_tag}")
    print(f"  Output root : {base_output_dir}")
    print(f"  Modalities  : {modalities}")
    print(f"  Tasks       : {tasks}")
    print(f"{'='*70}")

    failed: List[Tuple[str, str, str]] = []

    for modality in modalities:
        display = MODALITY_CONFIGS[modality]["display"]
        print(f"\n{'='*70}")
        print(f"  [Modality] {display}")
        print(f"{'='*70}")

        # ── Build combined MRI+PET manifest (ADNI3 AD + ADNI4 all) ──────────
        print(f"\n[{display}] Building samples manifest...")
        samples = build_modality_samples(modality)
        if samples.empty:
            print(f"[Warn] No samples found for {display}, skipping.")
            continue

        manifest_dir = base_output_dir / "_manifests" / run_tag / modality
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_csv = manifest_dir / "samples_manifest.csv"
        samples.to_csv(manifest_csv, index=False)
        print(f"  Manifest saved : {manifest_csv}")

        # ── Shared test split (stratified 3-class) ───────────────────────────
        print(f"\n[{display}] Building shared test split (test_ratio={args.test_ratio})...")
        test_subjects = build_shared_test_subjects(samples, args.test_ratio, args.seed)

        test_subjects_csv = manifest_dir / "shared_test_subjects.csv"
        pd.DataFrame({"Subject_ID": sorted(test_subjects)}).to_csv(test_subjects_csv, index=False)

        # ── Shared train/val/test split ──────────────────────────────────────
        print(f"\n[{display}] Building shared train/val/test split...")
        split_root = base_output_dir / "_splits" / run_tag / modality
        try:
            shared_split_csv = build_shared_subject_split(
                samples=samples,
                test_subjects=test_subjects,
                val_ratio=args.val_ratio,
                seed=args.seed,
                split_root=split_root,
            )
        except RuntimeError as exc:
            print(f"  [Error] {exc}")
            if not args.continue_on_error:
                return 1
            continue

        # ── Run one experiment per task ──────────────────────────────────────
        for task in tasks:
            out_dir = base_output_dir / modality / "multimodal_ot" / task / run_tag
            out_dir.mkdir(parents=True, exist_ok=True)
            exp_name = f"{modality}/{task}"
            print(f"\n[Run] {exp_name}  →  {out_dir}")

            if args.dry_run:
                print(f"[DryRun] would call run_single_task for {exp_name}")
                continue

            try:
                mri_ckpt = _resolve_ckpt(args.mri_brainmvp_ckpt, task=task, modality=modality)
                pet_ckpt = _resolve_ckpt(args.pet_brainmvp_ckpt, task=task, modality=modality)
            except FileNotFoundError as exc:
                failed.append((modality, task, str(exc)))
                print(f"[Fail] {exp_name}: {exc}")
                if not args.continue_on_error:
                    break
                continue

            import argparse as _ap
            task_args = _ap.Namespace(
                task=task,
                adni_phase="",
                samples_csv=str(manifest_csv),
                subject_split_csv=str(shared_split_csv),
                match_csv="",
                mri_root="",
                pet_root="",
                output_dir=str(out_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                depth=args.depth,
                height=args.height,
                width=args.width,
                feature_dim=args.feature_dim,
                encoder_backbone=args.encoder_backbone,
                mri_brainmvp_ckpt=mri_ckpt,
                pet_brainmvp_ckpt=pet_ckpt,
                brainmvp_trainable=args.brainmvp_trainable,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_target=args.lora_target,
                dropout=args.dropout,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                max_samples=args.max_samples,
                samples_per_class=args.samples_per_class,
                device=args.device,
                ot_weight=args.ot_weight,
                ot_epsilon=args.ot_epsilon,
                ot_iters=args.ot_iters,
                classwise_ot=args.classwise_ot,
                confusion_matrix_interval=args.confusion_matrix_interval,
                augment=args.augment,
            )

            try:
                run_single_task(task_args)
                print(f"[Done] {exp_name} → {out_dir}")
            except Exception as exc:
                failed.append((modality, task, str(exc)))
                print(f"[Fail] {exp_name}: {exc}")
                traceback.print_exc()
                if not args.continue_on_error:
                    break

        if failed and not args.continue_on_error:
            break

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    if not failed:
        print("  All experiments completed successfully.")
        return 0
    for modality, task, msg in failed:
        print(f"  [FAIL] [{modality}] task={task}: {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run AD/MCI/CN and AD/CN classification on FreeSurfer-preprocessed Abeta-PET and Tau-PET.

Data sources:
  AD:      ADNI3 + ADNI4 (both modalities)
  MCI, CN: ADNI4 only

The test split is shared across both tasks (AD-MCI-CN and AD-CN) within the same
modality, so the same held-out subjects are never seen during training for either task.

Output structure:
  <base-output-dir>/<modality>/<model>/<task>/<run-tag>/

Usage:
    python run_adni_freesurfer_pet_classification.py
    python run_adni_freesurfer_pet_classification.py --modality abeta_pet --models resnet3d
    python run_adni_freesurfer_pet_classification.py --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# ===========================================================================
# ── CONFIGURATION ───────────────────────────────────────────────────────────
#   All input paths, output paths, and default script names are collected here.
#   Edit this section to point to your data and choose where results are saved.
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

# ── Input dataset roots (raw ADNI PET series dirs and FreeSurfer output dirs) ─
ADNI3_PET_RAW_ROOT       = "/mnt/nvme_home/OT/datasets/ADNI3/PET/ADNI"
ADNI3_PET_PREPROC_ROOT   = "/mnt/nvme_home/OT/datasets/ADNI3/PET_preprocessed/ADNI"
ADNI4_PET_RAW_ROOT       = "/mnt/nvme_home/OT/datasets/ADNI4/PET/ADNI"
ADNI4_PET_PREPROC_ROOT   = "/mnt/nvme_home/OT/datasets/ADNI4/PET_preprocessed/ADNI"

# ── Output directory (all results, manifests, and split CSVs go here) ────────
DEFAULT_BASE_OUTPUT = "/mnt/nvme_home/OT/results/adni_pet_classification"

# ── Training script paths (resolved relative to this file's directory) ───────
DEFAULT_RESNET_SCRIPT    = "train_adni_pet_resnet3d.py"
DEFAULT_BRAINMVP_SCRIPT  = "train_adni_pet_brainmvp_uniformer.py"

# ── Default Python interpreter ────────────────────────────────────────────────
DEFAULT_PYTHON = "/home/prml/miniconda3/envs/OT/bin/python"

# ===========================================================================
# ── INTERNAL CONSTANTS (no need to edit below) ───────────────────────────────
# ===========================================================================

# Per-modality config (references the path constants above)
MODALITY_CONFIGS: Dict[str, Dict] = {
    "abeta_pet": {
        "display": "Abeta-PET",
        "sources": [
            {
                "phase":           "ADNI3",
                "csv":             ADNI3_ABETA_CSV,
                "raw_pet_root":    ADNI3_PET_RAW_ROOT,
                "preproc_pet_root": ADNI3_PET_PREPROC_ROOT,
                "class_filter":    {"AD"},   # ADNI3: only has AD labels
            },
            {
                "phase":           "ADNI4",
                "csv":             ADNI4_ABETA_CSV,
                "raw_pet_root":    ADNI4_PET_RAW_ROOT,
                "preproc_pet_root": ADNI4_PET_PREPROC_ROOT,
                "class_filter":    None,     # ADNI4: CN / MCI / AD all used
            },
        ],
    },
    "tau_pet": {
        "display": "Tau-PET",
        "sources": [
            {
                "phase":           "ADNI3",
                "csv":             ADNI3_TAU_CSV,
                "raw_pet_root":    ADNI3_PET_RAW_ROOT,
                "preproc_pet_root": ADNI3_PET_PREPROC_ROOT,
                "class_filter":    {"AD"},
            },
            {
                "phase":           "ADNI4",
                "csv":             ADNI4_TAU_CSV,
                "raw_pet_root":    ADNI4_PET_RAW_ROOT,
                "preproc_pet_root": ADNI4_PET_PREPROC_ROOT,
                "class_filter":    None,
            },
        ],
    },
}

# 3-class label map used for stratified test-split (superset of all tasks)
CLASS_3 = {"CN": 0, "MCI": 1, "AD": 2}

TASK_CLASS_MAPS: Dict[str, Dict[str, int]] = {
    "ad_mci_cn":    {"CN": 0, "MCI": 1, "AD": 2},
    "binary_ad_cn": {"CN": 0, "AD": 1},
}
TASKS = list(TASK_CLASS_MAPS.keys())


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_tracer(row: pd.Series) -> str:
    """Extract PET tracer from row, falling back to description parsing."""
    tracer = str(row.get("PET_PET_Tracer", row.get("PET_Tracer", ""))).strip().upper()
    if tracer and tracer not in ("NAN", ""):
        return tracer
    desc = str(row.get("PET_Description", "")).upper()
    for t in ("AV45", "FBB", "AV1451", "AV1415", "NAV4694", "PI2620", "MK6240", "FDG"):
        if t in desc:
            return t
    return "UNKNOWN"


def _map_to_preproc_path(
    series_dir: str,
    raw_pet_root: str,
    preproc_pet_root: str,
) -> Path:
    """Map raw ADNI PET series directory to PET-preprocessed NIfTI path.

    PET_preprocessed naming:
      <preproc_root>/<subject>/<series_desc>/<date>/<series_id>/<series_id>_preproc.nii.gz
    """
    series_path = Path(series_dir).resolve()
    raw_root    = Path(raw_pet_root).resolve()
    preproc_root = Path(preproc_pet_root).resolve()

    try:
        rel = series_path.relative_to(raw_root)
    except ValueError:
        # Fallback: anchor on PET/ADNI segment in the path
        parts = series_path.parts
        rel = None
        for i in range(len(parts) - 1):
            if parts[i].upper() == "PET" and parts[i + 1].upper() == "ADNI":
                rel = Path(*parts[i + 2:])
                break
        if rel is None:
            rel = Path(*series_path.parts[-4:]) if len(series_path.parts) >= 4 else Path(series_path.name)

    # rel = <subject>/<series_desc>/<date>/<series_id>
    return preproc_root / rel / f"{series_path.name}_preproc.nii.gz"


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def _load_phase_samples(
    csv_path: str,
    raw_pet_root: str,
    preproc_pet_root: str,
    phase: str,
    class_filter: Optional[Set[str]],
) -> pd.DataFrame:
    """Load one ADNI phase paired CSV and resolve FreeSurfer preprocessed paths."""
    df = pd.read_csv(csv_path)
    rows = []
    missing = 0

    for _, row in df.iterrows():
        rg = str(row.get("MRI_Research Group", "")).strip().upper()
        if not rg or rg == "NAN":
            continue
        if class_filter is not None and rg not in class_filter:
            continue

        series_dir = str(row.get("PET_Series_Dir", "")).strip()
        if not series_dir:
            continue

        tracer = _resolve_tracer(row)
        # nii = _map_to_preproc_path(series_dir, raw_pet_root, preproc_pet_root, tracer)
        nii = _map_to_preproc_path(series_dir, raw_pet_root, preproc_pet_root)
        if not nii.exists():
            missing += 1
            continue

        rows.append({
            "Subject_ID":      str(row.get("Subject ID", "")).strip(),
            "Research_Group":  rg,
            "PET_Date":        str(row.get("PET_Study Date", "")),
            "PET_Description": str(row.get("PET_Description", "")),
            "tracer":          tracer,
            "phase":           phase,
            "pet_path":        str(nii),
        })

    result = pd.DataFrame(rows)
    if missing:
        print(f"    [Warn] {missing}/{len(df)} rows had no FreeSurfer preprocessed file")
    return result


def build_modality_samples(modality: str) -> pd.DataFrame:
    """Combine ADNI3 (AD only) + ADNI4 (all classes) for the given modality."""
    cfg = MODALITY_CONFIGS[modality]
    all_parts = []

    for src in cfg["sources"]:
        phase   = src["phase"]
        cf      = {c.upper() for c in src["class_filter"]} if src["class_filter"] else None
        cf_label = f"{sorted(cf)}" if cf else "all"
        print(f"  [{phase}] classes={cf_label}  csv={Path(src['csv']).name}")
        part = _load_phase_samples(
            csv_path=src["csv"],
            raw_pet_root=src["raw_pet_root"],
            preproc_pet_root=src["preproc_pet_root"],
            phase=phase,
            class_filter=cf,
        )
        dist = part["Research_Group"].value_counts().to_dict() if not part.empty else {}
        n_subj = part["Subject_ID"].nunique() if not part.empty else 0
        print(f"    Resolved {len(part)} samples, {n_subj} subjects  dist={dist}")
        all_parts.append(part)

    combined = pd.concat(all_parts, ignore_index=True)
    combined = combined[combined["Subject_ID"] != ""].copy()
    combined = combined.drop_duplicates(subset=["pet_path"], keep="first").reset_index(drop=True)
    dist = combined["Research_Group"].value_counts().to_dict() if not combined.empty else {}
    n_subj = combined["Subject_ID"].nunique() if not combined.empty else 0
    print(f"  Combined: {len(combined)} samples, {n_subj} subjects  dist={dist}")
    return combined


# ---------------------------------------------------------------------------
# Split building  (single train / val / test — no KFold)
# ---------------------------------------------------------------------------

def _subject_label_table(samples: pd.DataFrame, class_map: Dict[str, int]) -> pd.DataFrame:
    """Return one row per subject with integer label for the given class map."""
    valid = samples[samples["Research_Group"].isin(class_map)].copy()
    valid["label"] = valid["Research_Group"].map(class_map)
    subj = valid.groupby("Subject_ID", as_index=False).agg(
        {"label": "first", "Research_Group": "first"}
    )
    return subj.reset_index(drop=True)


def build_shared_test_subjects(
    samples: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Set[str]:
    """Stratified test split over the 3-class universe; result is shared across tasks."""
    subj_df = _subject_label_table(samples, CLASS_3)
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
    """Create a single train/val/test subject-split CSV shared across all tasks.

    Uses the 3-class universe (CN, MCI, AD) so all tasks see the same subjects in
    train/val/test. Task-specific filtering happens at training time.

    Returns the path to the saved split CSV.
    The CSV has columns: Subject_ID, split  (values: train | val | test)
    """
    # Subject-level table from 3-class universe
    subj_df = _subject_label_table(samples, CLASS_3)

    test_df    = subj_df[subj_df["Subject_ID"].isin(test_subjects)].copy()
    trainval_df = subj_df[~subj_df["Subject_ID"].isin(test_subjects)].copy()

    if trainval_df.empty:
        raise RuntimeError("No train/val subjects left after test split.")

    # Stratified val split from the trainval pool
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

    train_set = set(train_ids)
    val_set   = set(val_ids)

    split_df = pd.concat(
        [
            trainval_df[trainval_df["Subject_ID"].isin(train_set)][["Subject_ID"]].assign(split="train"),
            trainval_df[trainval_df["Subject_ID"].isin(val_set)][["Subject_ID"]].assign(split="val"),
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
    print(f"  [Shared] split saved: train={n_train} val={n_val} test={n_test}  → {split_csv}")
    return split_csv


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _default_python() -> str:
    preferred = Path(DEFAULT_PYTHON)
    return str(preferred) if preferred.is_file() else sys.executable


def _run_one(cmd: List[str], dry_run: bool) -> int:
    print(f"\n[Run] {' '.join(shlex.quote(x) for x in cmd)}")
    if dry_run:
        return 0
    return int(subprocess.run(cmd).returncode)


def _token(value: object) -> str:
    text = str(value).strip().lower()
    return text.replace(".", "p").replace("-", "m").replace("+", "").replace("/", "_").replace(" ", "")


def _effective_run_tag(args: argparse.Namespace) -> str:
    """Build a human-readable run tag from key hyperparameters (or use --run-tag)."""
    if args.run_tag:
        return args.run_tag
    date_str = datetime.now().strftime("%Y%m%d")
    return (
        f"{date_str}_ep{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{_token(args.lr)}"
        f"_wd{_token(args.weight_decay)}"
        f"_seed{args.seed}"
    )


def _build_shared_training_args(args: argparse.Namespace) -> List[str]:
    return [
        "--epochs",       str(args.epochs),
        "--batch-size",   str(args.batch_size),
        "--lr",           str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--num-workers",  str(args.num_workers),
        "--depth",        str(args.depth),
        "--height",       str(args.height),
        "--width",        str(args.width),
        "--val-ratio",    str(args.val_ratio),
        "--test-ratio",   str(args.test_ratio),
        "--seed",         str(args.seed),
    ]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AD/MCI/CN and AD/CN classification on FreeSurfer-preprocessed "
            "Abeta-PET and Tau-PET.\n"
            "AD samples: ADNI3 + ADNI4.  MCI / CN samples: ADNI4 only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── What to run ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--modality", choices=["abeta_pet", "tau_pet", "all"], default="abeta_pet",
        help="Which PET modality to process.",
    )
    parser.add_argument(
        "--models", choices=["brainmvp", "resnet3d", "both"], default="resnet3d",
        help="Which model(s) to train.",
    )

    # ── Script / path overrides ──────────────────────────────────────────────
    parser.add_argument("--python",          type=str, default="",
                        help="Python interpreter to use (default: auto-detect OT env).")
    parser.add_argument("--resnet-script",   type=str, default=DEFAULT_RESNET_SCRIPT,
                        help="Path to the ResNet3D training script.")
    parser.add_argument("--resnet-backbone", type=str, nargs="+",
                        choices=["resnet34", "resnet50", "resnet101"],
                        default=["resnet50", "resnet101"],
                        help="ResNet backbone(s) to run. Defaults to all three.")
    parser.add_argument("--brainmvp-script", type=str, default=DEFAULT_BRAINMVP_SCRIPT,
                        help="Path to the BrainMVP training script.")
    parser.add_argument("--base-output-dir", type=str, default=DEFAULT_BASE_OUTPUT,
                        help="Root directory for all experiment outputs.")
    parser.add_argument("--run-tag",         type=str, default="",
                        help="Optional tag appended to output paths. Auto-generated from hyperparams if empty.")

    # ── Data split ───────────────────────────────────────────────────────────
    parser.add_argument("--val-ratio",  type=float, default=0.15,
                        help="Fraction of non-test subjects used for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Fraction of all subjects held out for testing.")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for all splits.")

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--weight-decay",type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--depth",       type=int,   default=128)
    parser.add_argument("--height",      type=int,   default=128)
    parser.add_argument("--width",       type=int,   default=128)

    # ── BrainMVP-specific ─────────────────────────────────────────────────────
    parser.add_argument("--pretrained",              type=str,   default="BrainMVP/BrainMVP_uniformer.pt",
                        help="Path to BrainMVP pretrained weights.")
    parser.add_argument("--peft-mode",               choices=["full", "lora"], default="full")
    parser.add_argument("--lora-rank",               type=int,   default=8)
    parser.add_argument("--lora-alpha",              type=float, default=16.0)
    parser.add_argument("--lora-dropout",            type=float, default=0.0)
    parser.add_argument("--dropout",                 type=float, default=0.2)
    parser.add_argument("--confusion-matrix-interval", type=int, default=0)

    # ── Execution control ─────────────────────────────────────────────────────
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep running other experiments if one fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    python          = args.python if args.python else _default_python()
    base_output_dir = Path(args.base_output_dir).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_tag         = _effective_run_tag(args)
    shared_train_args = _build_shared_training_args(args)

    modalities  = list(MODALITY_CONFIGS.keys()) if args.modality == "all" else [args.modality]
    run_brainmvp = args.models in ("brainmvp", "both")
    run_resnet   = args.models in ("resnet3d", "both")

    print(f"\n{'='*70}")
    print(f"  Run tag        : {run_tag}")
    print(f"  Output root    : {base_output_dir}")
    print(f"  Modalities     : {modalities}")
    resnet_label = " ".join(args.resnet_backbone) if run_resnet else ""
    print(f"  Models         : {'brainmvp ' if run_brainmvp else ''}{resnet_label}")
    print(f"  test_ratio={args.test_ratio}  val_ratio={args.val_ratio}  seed={args.seed}")
    print(f"{'='*70}")

    failed: List[Tuple[str, str, str, int]] = []

    for modality in modalities:
        display = MODALITY_CONFIGS[modality]["display"]
        print(f"\n{'='*70}")
        print(f"  [Modality] {display}")
        print(f"{'='*70}")

        # ── Build samples manifest ───────────────────────────────────────────
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
        print(f"  Test subjects  : {test_subjects_csv}")

        # ── Shared train/val/test split (3-class universe, used by all tasks) ───
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
        except RuntimeError as e:
            print(f"  [Error] {e}")
            if not args.continue_on_error:
                return 1
            continue

        # All tasks use the same split (task-specific filtering happens at training)
        split_csv_by_task = {task: shared_split_csv for task in TASKS}

        # ── Build experiment list ────────────────────────────────────────────
        experiments: List[Dict] = []

        if run_resnet:
            resnet_script = Path(args.resnet_script).expanduser().resolve()
            if not resnet_script.is_file():
                raise FileNotFoundError(f"ResNet3D script not found: {resnet_script}")
            for backbone in args.resnet_backbone:
                for task in TASKS:
                    experiments.append({
                        "name":      f"{backbone}_{task}_{modality}",
                        "script":    str(resnet_script),
                        "model_dir": backbone,
                        "task":      task,
                        "extra":     [
                            "--backbone", backbone,
                            "--confusion-matrix-interval", str(args.confusion_matrix_interval),
                        ],
                    })

        if run_brainmvp:
            brainmvp_script = Path(args.brainmvp_script).expanduser().resolve()
            if not brainmvp_script.is_file():
                raise FileNotFoundError(f"BrainMVP script not found: {brainmvp_script}")
            brainmvp_extra = [
                "--pretrained",   args.pretrained,
                "--peft-mode",    args.peft_mode,
                "--lora-rank",    str(args.lora_rank),
                "--lora-alpha",   str(args.lora_alpha),
                "--lora-dropout", str(args.lora_dropout),
                "--dropout",      str(args.dropout),
            ]
            for task in TASKS:
                experiments.append({
                    "name":      f"brainmvp_{task}_{modality}",
                    "script":    str(brainmvp_script),
                    "model_dir": "brainmvp_uniformer",
                    "task":      task,
                    "extra":     brainmvp_extra,
                })

        # ── Run experiments ──────────────────────────────────────────────────
        for exp in experiments:
            task = exp["task"]
            if task not in split_csv_by_task:
                print(f"[Skip] No split CSV for task={task}, skipping {exp['name']}.")
                continue

            split_csv = split_csv_by_task[task]
            out_dir   = base_output_dir / modality / exp["model_dir"] / task / run_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                python,
                exp["script"],
                "--task",              task,
                "--output-dir",        str(out_dir),
                "--samples-csv",       str(manifest_csv),
                "--subject-split-csv", str(split_csv),
                "--adni-phase",        "ADNI4",  # only affects default path resolution (unused with --samples-csv)
                *shared_train_args,
                *exp["extra"],
            ]
            rc = _run_one(cmd=cmd, dry_run=bool(args.dry_run))
            if rc != 0:
                failed.append((modality, exp["name"], task, rc))
                print(f"[Fail] {exp['name']} task={task} exit={rc}")
                if not args.continue_on_error:
                    break
            else:
                print(f"[Done] {exp['name']} → {out_dir}")

        if failed and not args.continue_on_error:
            break

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    if not failed:
        print("  All experiments completed successfully.")
        return 0
    for modality, name, task, rc in failed:
        print(f"  [FAIL] [{modality}] {name} task={task} exit_code={rc}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

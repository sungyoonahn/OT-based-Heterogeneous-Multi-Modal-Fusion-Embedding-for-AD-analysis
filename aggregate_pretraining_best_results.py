#!/usr/bin/env python3
"""Summarize best 3D ResNet pretraining runs into a single CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Accept decimals or scientific notation.
NUMBER_PATTERN = r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"

CSV_COLUMNS: List[str] = [
    "setup",
    "modality",
    "model_depth",
    "data_split",
    "dropout",
    "pretrained",
    "attention_target",
    "best_val_loss",
    "best_epoch",
    "val_acc",
    "precision",
    "recall",
    "f1_score",
    "specificity",
]

NUMERIC_COLUMNS = [
    "best_val_loss",
    "best_epoch",
    "val_acc",
    "precision",
    "recall",
    "f1_score",
    "specificity",
]


def parse_results_file(path: Path, default_modality: str) -> Optional[Dict[str, str]]:
    """Extracts the best metrics block from a single results.txt file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return None

    modality, model_depth, data_split, extras = parse_setup_fields(path.parent.name, default_modality)
    best_val_loss = _search(rf"Best Validation Loss:\s*({NUMBER_PATTERN})", text)
    metrics_match = re.search(
        rf"Best (?:Epoch|Metrics\s*\|\s*Epoch):\s*(\d+)\s+Acc:\s*({NUMBER_PATTERN})\s+Precision:\s*({NUMBER_PATTERN})\s+"
        rf"Recall:\s*({NUMBER_PATTERN})\s+F1:\s*({NUMBER_PATTERN})\s+Specificity:\s*({NUMBER_PATTERN})",
        text,
    )

    if not (best_val_loss and metrics_match):
        return None

    row = {
        "setup": path.parent.name,
        "modality": modality,
        "model_depth": model_depth,
        "data_split": data_split,
        "best_val_loss": best_val_loss,
        "best_epoch": metrics_match.group(1),
        "val_acc": metrics_match.group(2),
        "precision": metrics_match.group(3),
        "recall": metrics_match.group(4),
        "f1_score": metrics_match.group(5),
        "specificity": metrics_match.group(6),
    }
    for key in ("dropout", "pretrained", "attention_target"):
        row[key] = extras.get(key, "")
    return row


def _search(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def parse_setup_fields(
    setup_name: str, default_modality: str = ""
) -> Tuple[str, str, str, Dict[str, str]]:
    """
    Split a setup string into modality/depth/split components and optional extras like dropout,
    pretraining flag, and attention targets. Falls back gracefully if the layout is unfamiliar.
    """
    extras: Dict[str, str] = {"dropout": "", "pretrained": "", "attention_target": ""}

    attn_pattern = re.match(
        r"^mdepth(?P<depth>\d+)_drop(?P<dropout>[^_]+)_(?P<split>all|balanced)_(?P<pretrain>(?:with|no)_pretrain)"
        r"(?:_(?P<attn>.+))?$",
        setup_name,
    )
    if attn_pattern:
        depth = attn_pattern.group("depth")
        split = attn_pattern.group("split")
        dropout = attn_pattern.group("dropout")
        pretrain = attn_pattern.group("pretrain")
        attn_raw = attn_pattern.group("attn") or ""

        extras["dropout"] = dropout
        extras["pretrained"] = pretrain
        extras["attention_target"] = _normalize_attention_target(attn_raw)

        modality = default_modality.strip() or "mdepth"
        return modality, depth, split, extras

    match = re.match(r"^(?P<modality>[^_]+)_depth(?P<depth>\d+)_(?P<split>.+)$", setup_name)
    if match:
        return match.group("modality"), match.group("depth"), match.group("split"), extras

    match = re.match(r"^depth(?P<depth>\d+)_(?P<split>.+)$", setup_name)
    if match:
        modality = default_modality.strip()
        return modality, match.group("depth"), match.group("split"), extras

    depth_match = re.search(r"depth(?P<depth>\d+)", setup_name)
    depth = depth_match.group("depth") if depth_match else ""

    tokens = setup_name.split("_", 1)
    first_token = tokens[0]
    data_split = tokens[1] if len(tokens) > 1 else ""

    letters_match = re.match(r"([A-Za-z]+)", first_token)
    modality_from_name = letters_match.group(1) if letters_match else first_token
    modality = default_modality.strip() or modality_from_name or setup_name

    return modality, depth, data_split, extras


def _normalize_attention_target(attn_suffix: str) -> str:
    if not attn_suffix:
        return "none"
    if "mri_pet_attn" in attn_suffix:
        return "mri_pet"
    if "mri_attn" in attn_suffix:
        return "mri"
    if "pet_attn" in attn_suffix:
        return "pet"
    return attn_suffix


def collect_best_results(results_dir: Path, default_modality: str) -> List[Dict[str, str]]:
    """Walk the directory and collect parsed rows."""
    rows: List[Dict[str, str]] = []
    for result_file in sorted(results_dir.rglob("results.txt")):
        parsed = parse_results_file(result_file, default_modality)
        if parsed is None:
            print(f"[WARN] Skipping {result_file} (missing best metrics block)", file=sys.stderr)
            continue
        rows.append(parsed)
    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_excel(rows: List[Dict[str, str]], output_path: Path) -> None:
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def resolve_default_results_dir(script_path: Path) -> Path:
    project_root = script_path.parent
    return project_root / "results" / "3DResNet_pretraining"


def infer_default_output_stem(results_dir: Path) -> str:
    mapping = {
        "3DResNet_pretraining": "best_pretraining_results",
        "MRI_PET_mmfusion_sweeps": "best_mmfusion_results",
        "MRI_PET_OT_attention": "best_mri_pet_ot_attention_results",
    }
    return mapping.get(results_dir.name, f"best_{results_dir.name}_results")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="Aggregate best metrics from 3D ResNet pretraining runs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=resolve_default_results_dir(script_path),
        help="Path to the directory that contains 3DResNet_pretraining subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save the aggregated CSV. Defaults to a file inside --results-dir.",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        help="Where to save the aggregated Excel file. Defaults to a file inside --results-dir.",
    )
    parser.add_argument(
        "--default-modality",
        type=str,
        default="",
        help="Value to populate the modality column when it cannot be inferred from the setup name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect_best_results(results_dir, args.default_modality)
    if not rows:
        print(f"[WARN] No results.txt files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    stem = infer_default_output_stem(results_dir)
    csv_path = args.output.expanduser() if args.output else results_dir / f"{stem}.csv"
    xlsx_path = args.excel_output.expanduser() if args.excel_output else results_dir / f"{stem}.xlsx"
    write_csv(rows, csv_path)
    write_excel(rows, xlsx_path)
    print(f"Wrote {len(rows)} rows to {csv_path} and {xlsx_path}")


if __name__ == "__main__":
    main()

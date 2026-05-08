#!/usr/bin/env python3
"""
Combine sweep CSVs

Inputs:
  - All files matching: sweep_table_*.csv
    - By default, searched recursively under <input_dir>
      (e.g., results/sweep_*/sweep_table_*.csv)

Outputs (written to output_dir):
  - combined_all_sweeps_tidy.csv

python3 analysis.py --input_dir results --output_dir results/analysis_summary
"""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_combine(input_dir: Path, recursive: bool = True) -> pd.DataFrame:
    
    """
    Load all sweep_table_*.csv files from input_dir and combine into
    a single DataFrame and add 'sweep' column to identify which variable we are sweeping on.
    """
    
    if recursive:
        paths = sorted(str(p) for p in input_dir.rglob("sweep_table_*.csv"))
    else:
        paths = sorted(glob(str(input_dir / "sweep_table_*.csv")))

    if not paths:
        hint = (
            "If your sweeps are in subfolders (e.g., results/sweep_*/), use --recursive."
            if not recursive else
            "Make sure the folder contains sweep_table_*.csv files."
        )
        raise FileNotFoundError(f"No sweep_table_*.csv files found under {input_dir}. {hint}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df.insert(0, "source_file", str(Path(p).resolve()))
        # infer sweep name from filename: sweep_table_<sweep>_<timestamp>.csv
        name = Path(p).stem.replace("sweep_table_", "")
        parts = name.rsplit("_", 1)
        sweep = parts[0] if len(parts) == 2 else name
        df.insert(0, "sweep", sweep)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Order columns 'sweep' first, then 'value', then everything else
    cols = list(all_df.columns)
    if "value" in cols:
        cols = ["sweep", "value"] + [c for c in cols if c not in ("sweep", "value")]
        all_df = all_df[cols]

    return all_df

def main():
    parser = argparse.ArgumentParser(description="Combine sweep CSVs.")
    parser.add_argument("--input_dir", type=str, default=".", help="Folder containing sweep_table_*.csv")
    parser.add_argument("--output_dir", type=str, default=".", help="Folder to write outputs")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for sweep_table_*.csv recursively under input_dir (recommended for results/sweep_*/ folders)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only search for sweep_table_*.csv directly inside input_dir",
    )
    parser.set_defaults(recursive=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & combine
    all_df = load_and_combine(input_dir, recursive=args.recursive)

    # 2) Save combined
    combined_csv = output_dir / "combined_all_sweeps_tidy.csv"
    all_df.to_csv(combined_csv, index=False)

if __name__ == "__main__":
    main()


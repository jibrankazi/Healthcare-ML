import argparse
import json
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description="Generate LaTeX results table")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open(cfg["outputs"]["results_json"]) as f:
        res = json.load(f)

    out_path = Path(cfg["outputs"]["latex_table"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = (
        "%% Auto-generated -- do not edit by hand\n"
        "\\begin{tabular}{l r}\n"
        "\\hline\n"
        "Metric & Value \\\\ \\hline\n"
        f"AUROC & {res['AUROC']:.4f} \\\\\n"
        f"AUPRC & {res['AUPRC']:.4f} \\\\ \\hline\n"
        "\\end{tabular}\n"
    )

    out_path.write_text(table, encoding="utf-8")
    print("Wrote LaTeX table to", out_path)


if __name__ == "__main__":
    main()

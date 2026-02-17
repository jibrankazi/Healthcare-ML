"""Generate a LaTeX comparison table from multi-model results.json."""

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

    models = res["models"]

    lines = [
        "%% Auto-generated -- do not edit by hand",
        "\\begin{tabular}{l c c c}",
        "\\hline",
        "Model & AUROC [95\\% CI] & AUPRC [95\\% CI] & Brier \\\\ \\hline",
    ]

    for name, m in models.items():
        auroc = m["AUROC"]
        auroc_ci = m["AUROC_CI"]
        auprc = m["AUPRC"]
        auprc_ci = m["AUPRC_CI"]
        brier = m["Brier"]
        lines.append(
            f"{name} & {auroc:.4f} [{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}] "
            f"& {auprc:.4f} [{auprc_ci[0]:.3f}, {auprc_ci[1]:.3f}] "
            f"& {brier:.4f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    table = "\n".join(lines) + "\n"
    out_path.write_text(table, encoding="utf-8")
    print("Wrote LaTeX table to", out_path)


if __name__ == "__main__":
    main()

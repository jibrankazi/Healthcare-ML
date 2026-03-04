"""Cross-platform pipeline runner: train -> evaluate -> latex."""
import subprocess
import sys


STEPS = [
    ("train", [sys.executable, "-m", "hchealth.train", "--config"]),
    ("evaluate", [sys.executable, "-m", "hchealth.evaluate", "--config"]),
    ("latex", [sys.executable, "-m", "hchealth.to_latex", "--config"]),
]


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run the full Healthcare-ML pipeline")
    ap.add_argument(
        "--config",
        default="configs/clinical_demo.yaml",
        help="Path to YAML config file",
    )
    args = ap.parse_args()

    for name, cmd in STEPS:
        full_cmd = cmd + [args.config]
        print(f"\n{'='*40}")
        print(f"  Step: {name}")
        print(f"{'='*40}")
        result = subprocess.run(full_cmd)
        if result.returncode != 0:
            print(f"Step '{name}' failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*40}")
    print("  Pipeline complete!")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()

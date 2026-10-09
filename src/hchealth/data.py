"""Reproducible data selection for clinical *research* demos.

The sklearn Wisconsin Breast Cancer dataset is a real public research dataset
(not a clinical-deployment feed). If a user requests a Hugging Face dataset,
never silently replace it with a different source after a download error.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_tabular(hf_id="none", target_column="target"):
    """Return stratified 80/20 splits, rejecting unintended source substitutions.

    `hf_id="none"` explicitly selects sklearn's Wisconsin Breast Cancer data.
    Otherwise Hugging Face retrieval must succeed and the expected target must
    exist; the project must not report another dataset as the requested one.
    """
    if hf_id == "none":
        bunch = load_breast_cancer(as_frame=True)
        X = bunch.frame.drop(columns=["target"])
        y = bunch.frame["target"]
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_id, split="train")
            df = ds.to_pandas()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load requested Hugging Face dataset {hf_id!r}; "
                "refusing to substitute Wisconsin Breast Cancer data."
            ) from exc
        if target_column not in df.columns:
            raise ValueError(
                f"Target {target_column!r} absent from requested dataset {hf_id!r}"
            )
        y = df[target_column]
        X = df.drop(columns=[target_column])
        if X.empty or y.isna().any():
            raise ValueError("Requested dataset has no usable features or missing target")

    return train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

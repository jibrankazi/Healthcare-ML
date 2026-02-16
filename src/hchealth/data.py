from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_tabular(hf_id="none", target_column="target"):
    """Load a tabular dataset and return train/test splits.

    Tries Hugging Face Datasets API first; falls back to sklearn breast cancer.
    Returns (X_train, X_test, y_train, y_test).
    """
    if hf_id != "none":
        try:
            from datasets import load_dataset

            ds = load_dataset(hf_id, split="train")
            df = ds.to_pandas()
            if target_column not in df.columns:
                target_column = df.columns[-1]
            y = df[target_column]
            X = df.drop(columns=[target_column])
        except Exception:
            print(f"Could not load '{hf_id}' from HF, falling back to breast cancer dataset.")
            bunch = load_breast_cancer(as_frame=True)
            X = bunch.frame.drop(columns=["target"])
            y = bunch.frame["target"]
    else:
        bunch = load_breast_cancer(as_frame=True)
        X = bunch.frame.drop(columns=["target"])
        y = bunch.frame["target"]

    return train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

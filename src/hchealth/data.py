from datasets import load_dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
def load_tabular(hf_id: str = 'none', target_column: str = 'target'):
    if hf_id != 'none':
        ds = load_dataset(hf_id, split='train')
        df = ds.to_pandas()
        if target_column not in df.columns:
            target_column = df.columns[-1]
        y = df[target_column]; X = df.drop(columns=[target_column])
    else:
        b = load_breast_cancer(as_frame=True); X = b.frame.drop(columns=['target']); y = b.frame['target']
    return train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

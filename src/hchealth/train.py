import argparse, os, json, yaml, numpy as np, random
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from hchealth.data import load_tabular
def set_seed(s): random.seed(s); np.random.seed(s)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); set_seed(cfg.get('seed', 42))
    out = cfg['outputs']['run_dir']; os.makedirs(out, exist_ok=True)
    X_train, y_train, X_test, y_test = load_tabular(cfg['data']['hf_dataset_id'], cfg['data']['target_column'])
    model = Pipeline([('scaler', StandardScaler(with_mean=False)), ('rf', RandomForestClassifier(
        n_estimators=cfg['model']['n_estimators'], max_depth=cfg['model']['max_depth'], random_state=0
    ))])
    model.fit(X_train, y_train)
    import joblib; joblib.dump(model, os.path.join(out, 'model.joblib'))
    json.dump({'n_features': int(X_train.shape[1]), 'n_train': int(len(y_train))}, open(os.path.join(out, 'train_meta.json'), 'w'))
    print('Training complete.')
if __name__ == '__main__': main()

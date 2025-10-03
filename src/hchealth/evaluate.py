import argparse, os, json, yaml, numpy as np, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from hchealth.data import load_tabular
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); out = cfg['outputs']['run_dir']; os.makedirs(out, exist_ok=True)
    import joblib; model = joblib.load(os.path.join(out, 'model.joblib'))
    X_train, y_train, X_test, y_test = load_tabular(cfg['data']['hf_dataset_id'], cfg['data']['target_column'])
    proba = model.predict_proba(X_test)[:,1]
    auroc = float(roc_auc_score(y_test, proba)); auprc = float(average_precision_score(y_test, proba))
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    results = {'AUROC': auroc, 'AUPRC': auprc, 'n_test': int(len(y_test))}
    json.dump(results, open(cfg['outputs']['results_json'], 'w'), indent=2)
    os.makedirs(cfg['outputs']['fig_dir'], exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.grid(True)
    plt.savefig(os.path.join(cfg['outputs']['fig_dir'], 'roc.png'), dpi=200)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    plt.figure(); plt.plot(recall, precision); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.grid(True)
    plt.savefig(os.path.join(cfg['outputs']['fig_dir'], 'pr.png'), dpi=200)
    plt.figure(); plt.plot(mean_pred, frac_pos, marker='o'); plt.plot([0,1],[0,1],'--'); plt.xlabel('Mean predicted'); plt.ylabel('Fraction positive'); plt.title('Calibration'); plt.grid(True)
    plt.savefig(os.path.join(cfg['outputs']['fig_dir'], 'calibration.png'), dpi=200)
    print('Results:', results)
if __name__ == '__main__': main()

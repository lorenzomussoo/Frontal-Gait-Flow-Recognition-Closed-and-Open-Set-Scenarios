import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc
from joblib import dump, load
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import box
from scipy.interpolate import interp1d
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models"
RESULTS_DIR = "results/svm_slope/open_set"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_ALL_VALID_AS_KNOWN = False
PERFORM_GRID_SEARCH = False 

NUM_KNOWN_SUBJECTS = 15
VIDEO_CUT_INDEX = 73728 
SEED = 42
np.random.seed(SEED)

def get_subject_pools():
    if not os.path.exists(FEATURES_ROOT): return None, None
    all_subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    valid_subjects = []
    for subj in all_subjects:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        if not (os.path.exists(os.path.join(subj_path, "FirstRun")) and 
                os.path.exists(os.path.join(subj_path, "SecondRun")) and
                os.path.exists(os.path.join(subj_path, "ThirdRun"))):
            continue
            
        is_complete = True
        for run_name in ["FirstRun", "SecondRun", "ThirdRun"]:
            run_dir = os.path.join(subj_path, run_name)
            slope_count = 0
            for root, dirs, files in os.walk(run_dir):
                dirs.sort()
                root_lower = root.lower()
                if 'debug' in root_lower or "slope" not in root_lower: continue
                for f in sorted(files):
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        slope_count += 1
            if slope_count < 6:
                is_complete = False; break
        if is_complete: valid_subjects.append(subj)

    incomplete_subjects = [s for s in all_subjects if s not in valid_subjects]
    
    if USE_ALL_VALID_AS_KNOWN:
        known_subjects = sorted(valid_subjects)
        unknown_subjects = sorted(incomplete_subjects)
    else:
        np.random.shuffle(valid_subjects)
        known_subjects = sorted(valid_subjects[:NUM_KNOWN_SUBJECTS])
        unknown_valid = valid_subjects[NUM_KNOWN_SUBJECTS:]
        unknown_subjects = sorted(unknown_valid + incomplete_subjects)
        
    return known_subjects, unknown_subjects

def load_data(known_subs, unknown_subs):
    X_train, y_train_str = [], []
    X_genuine, y_genuine_str = [], [] 
    X_impostor = [] 
    
    for subj in known_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            dirs.sort()
            root_lower = root.lower()
            if 'debug' in root_lower or "slope" not in root_lower: continue
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                    if len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
                    else: continue
                except: continue
                
                if 'FirstRun' in root or 'SecondRun' in root:
                    X_train.append(vec)
                    y_train_str.append(subj)
                elif 'ThirdRun' in root:
                    X_genuine.append(vec)
                    y_genuine_str.append(subj)

    for subj in unknown_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            dirs.sort()
            root_lower = root.lower()
            if 'debug' in root_lower or "slope" not in root_lower: continue
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                    if len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
                    else: continue
                except: continue
                X_impostor.append(vec)

    return (np.nan_to_num(X_train), np.array(y_train_str), 
            np.nan_to_num(X_genuine), np.array(y_genuine_str), 
            np.nan_to_num(X_impostor))

def calculate_metrics(probs_genuine, y_genuine, probs_impostor, thresholds):
    TG = len(probs_genuine)
    TI = len(probs_impostor)
    DIR_list, FAR_list, FRR_list = [], [], []
    
    for t in thresholds:
        DI, FA = 0, 0
        for i in range(TG):
            if np.max(probs_genuine[i]) >= t and np.argmax(probs_genuine[i]) == y_genuine[i]:
                DI += 1
        for i in range(TI):
            if np.max(probs_impostor[i]) >= t:
                FA += 1
                
        DIR_list.append(DI / TG if TG > 0 else 0)
        FAR_list.append(FA / TI if TI > 0 else 0)
        FRR_list.append(1 - (DI / TG) if TG > 0 else 1) 
        
    return np.array(DIR_list), np.array(FAR_list), np.array(FRR_list)

def main():
    console.print(Panel.fit("[bold magenta]OPEN SET PROTOCOL: SVM (Slope)[/bold magenta]"))
    known, unknown = get_subject_pools()
    if not known: return
    
    console.print(f"Gallery (Known): {len(known)} subjects")
    console.print(f"Impostors (Unknown): {len(unknown)} subjects")

    X_train, y_train_str, X_gen, y_gen_str, X_imp = load_data(known, unknown)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_gen = le.transform(y_gen_str)

    if USE_ALL_VALID_AS_KNOWN:
        model_name = "svm_slope_closed_set.joblib"
        params_name = "best_params_svm_slope_closed_set.json"
    else:
        model_name = "svm_slope_open_set.joblib"
        params_name = "best_params_svm_slope_open_set.json"
        
    model_path = os.path.join(MODEL_DIR, model_name)
    params_path = os.path.join(MODEL_DIR, params_name)

    is_grid_search = PERFORM_GRID_SEARCH and not USE_ALL_VALID_AS_KNOWN

    if not is_grid_search and os.path.exists(model_path):
        console.print(f"[bold green]Loading existing model: {model_name}...[/bold green]")
        model = load(model_path)
    else:
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()), 
            ('svm', SVC(probability=True, random_state=SEED))
        ])

        if is_grid_search:
            console.print("[bold yellow]Performing Grid Search...[/bold yellow]")
            param_grid = [
                {
                    'scaler': [StandardScaler()], 
                    'pca': ['passthrough', PCA(n_components=0.95)], 
                    'svm__C': [0.1, 1.0, 10.0], 
                    'svm__kernel': ['linear', 'rbf']
                }
            ]   

            search = GridSearchCV(base_pipeline, param_grid, cv=3, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            
            clean_params = {}
            for k, v in search.best_params_.items():
                if k == 'pca' and isinstance(v, PCA): 
                    clean_params['pca_status'] = 'active'
                    clean_params['pca__n_components'] = v.n_components
                elif k == 'pca' and v == 'passthrough': 
                    clean_params['pca_status'] = 'passthrough'
                elif k != 'scaler': 
                    clean_params[k] = v
            with open(params_path, 'w') as f: 
                json.dump(clean_params, f)
            dump(model, model_path)
            console.print(f"[bold green]Model and Best Params saved to {model_name} and {params_name}![/bold green]")
        else:
            console.print("[bold yellow]Training model with saved/default parameters...[/bold yellow]")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f: loaded_params = json.load(f)
                if loaded_params.get('pca_status') == 'passthrough' or loaded_params.get('pca') == 'passthrough':
                    base_pipeline.set_params(pca='passthrough')
                else: 
                    base_pipeline.set_params(pca=PCA(n_components=loaded_params.get('pca__n_components', 0.95)))
                svm_params = {k: v for k, v in loaded_params.items() if k.startswith('svm__')}
                base_pipeline.set_params(scaler=StandardScaler(), **svm_params)
            else:
                base_pipeline.set_params(scaler=StandardScaler(), pca='passthrough', svm__C=1.0, svm__kernel='linear')
            base_pipeline.fit(X_train, y_train)
            model = base_pipeline
            dump(model, model_path)
            console.print(f"[bold green]Model saved to {model_name}![/bold green]")

    console.print(f"Testing on {len(X_gen)} Genuine Probes and {len(X_imp)} Impostor Probes...")
    probs_gen = model.predict_proba(X_gen)
    probs_imp = model.predict_proba(X_imp)

    thresholds = np.linspace(0.0, 1.0, 200)
    DIR, FAR, FRR = calculate_metrics(probs_gen, y_gen, probs_imp, thresholds)

    def eer_func(t):
        return interp1d(thresholds, FAR)(t) - interp1d(thresholds, FRR)(t)

    eer_threshold = brentq(eer_func, 0.0, 1.0)
    eer_value = float(interp1d(thresholds, FAR)(eer_threshold))
    dir_at_eer = float(interp1d(thresholds, DIR)(eer_threshold))

    sort_idx = np.argsort(FAR)
    roc_auc = auc(FAR[sort_idx], DIR[sort_idx])

    table = Table(title="Open Set Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Equal Error Rate (EER)", f"{eer_value*100:.2f}%")
    table.add_row("Balance Threshold (EER)", f"{eer_threshold:.2f}")
    table.add_row("ROC AUC", f"{roc_auc:.4f}")
    console.print(table)

    suffix = "_19_known" if USE_ALL_VALID_AS_KNOWN else "_15_known"

    plt.figure(figsize=(10, 6))
    plt.plot(FAR, DIR, label='Watchlist ROC', color='purple', linewidth=2)
    plt.plot(eer_value, dir_at_eer, 'ro', markersize=8, label=f'EER Point ({eer_value*100:.1f}%)')
    plt.title('Watchlist ROC (Open Set) - SVM Slope')
    plt.xlabel('False Alarm Rate (FAR)')
    plt.ylabel('Detect & Identify Rate (DIR)')
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'roc_watchlist_svm{suffix}.png'), dpi=300)
    plt.close()

    with open(os.path.join(RESULTS_DIR, f"open_set_report{suffix}.txt"), "w") as f:
        f.write("========================================================\n")
        f.write("         OPEN SET WATCHLIST REPORT: SVM Slope\n")
        f.write("========================================================\n")
        f.write(f"Mode: {'All 19 Valid Subjects' if USE_ALL_VALID_AS_KNOWN else '15 Valid Subjects Subset'}\n")
        f.write(f"Known Subjects in Gallery: {len(known)}\n")
        f.write(f"Unknown Subjects (Impostors): {len(unknown)}\n")
        f.write(f"Genuine Probe Attempts (TG): {len(X_gen)}\n")
        f.write(f"Impostor Probe Attempts (TI): {len(X_imp)}\n\n")
        
        f.write("--- MAIN METRICS ---\n")
        f.write(f"EER (Equal Error Rate): {eer_value*100:.2f}%\n")
        f.write(f"Balance Threshold for EER: {eer_threshold:.2f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("--- DETAILED THRESHOLD ANALYSIS (DIR, FAR, FRR) ---\n")
        f.write(f"Note: FRR (False Reject Rate) = 100% - DIR\n\n")
        f.write(f"{'Threshold':<12} | {'DIR (Detect & ID)':<20} | {'FAR (False Alarm)':<20} | {'FRR (False Reject)':<20}\n")
        f.write("-" * 80 + "\n")
        
        target_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for tt in target_thresholds:
            closest_idx = np.argmin(np.abs(thresholds - tt))
            f.write(f"{thresholds[closest_idx]:<12.2f} | {DIR[closest_idx]*100:<18.2f}% | {FAR[closest_idx]*100:<18.2f}% | {FRR[closest_idx]*100:<18.2f}%\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{eer_threshold:<12.2f} | {dir_at_eer*100:<18.2f}% | {eer_value*100:<18.2f}% | {eer_value*100:<18.2f}% <-- EER POINT\n\n")

        f.write("--- APPLICATION NEEDS (OPERATIONAL POINTS) ---\n")
        for target_far in [0.01, 0.05, 0.10]:
            idx = np.where(FAR <= target_far)[0]
            if len(idx) > 0:
                best_idx = idx[0]
                f.write(f"- High Security (Max FAR {target_far*100}%): Threshold >= {thresholds[best_idx]:.2f}  -->  DIR = {DIR[best_idx]*100:.2f}%, FRR = {FRR[best_idx]*100:.2f}%\n")

    console.print(f"[bold green]Analysis saved in {RESULTS_DIR}[/bold green]")

if __name__ == "__main__":
    main()
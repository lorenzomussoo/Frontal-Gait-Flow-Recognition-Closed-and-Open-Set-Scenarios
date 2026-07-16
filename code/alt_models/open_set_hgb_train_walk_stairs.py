import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from scipy.interpolate import interp1d
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models/alt_models"
RESULTS_DIR = "results/alt_models/hgb_walk_stairs/masked_open_set"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

PERFORM_GRID_SEARCH = False 

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
            walk_count, stairs_count = 0, 0
            for root, dirs, files in os.walk(run_dir):
                root_lower = root.lower()
                if 'debug' in root_lower or "_backup" in root_lower or "slope" in root_lower: 
                    continue
                for f in files:
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        f_l = f.lower()
                        if 'walk' in f_l: walk_count += 1
                        elif 'stairs' in f_l or 'up' in f_l or 'down' in f_l: stairs_count += 1
            if walk_count < 6 or stairs_count < 6:
                is_complete = False; break
        if is_complete: valid_subjects.append(subj)

    incomplete_subjects = [s for s in all_subjects if s not in valid_subjects]
    return sorted(valid_subjects), sorted(incomplete_subjects)

def load_data(valid_subs, incomplete_subs):
    X_train, y_train_str = [], []
    X_gen, y_gen_str = [], []
    X_imp = []
    
    for subj in valid_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            root_lower = root.lower()
            if 'debug' in root_lower or "_backup" in root_lower or "slope" in root_lower: continue
                
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                except: continue
                
                if 'FirstRun' in root or 'SecondRun' in root:
                    X_train.append(vec)
                    y_train_str.append(subj)
                elif 'ThirdRun' in root:
                    X_gen.append(vec)
                    y_gen_str.append(subj)
                    
    for subj in incomplete_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            root_lower = root.lower()
            if 'debug' in root_lower or "_backup" in root_lower or "slope" in root_lower: continue
                
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                except: continue
                X_imp.append(vec)
                
    return (np.nan_to_num(X_train), np.array(y_train_str), 
            np.nan_to_num(X_gen), np.array(y_gen_str), 
            np.nan_to_num(X_imp))

def build_or_train_pipeline(X_train, y_train):
    params_path = os.path.join(MODEL_DIR, "best_params_hgb_walk_stairs.json")
    
    if PERFORM_GRID_SEARCH:
        console.print("[yellow]Running Grid Search Optimization for HGB...[/yellow]")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('hgb', HistGradientBoostingClassifier(random_state=SEED, early_stopping=False))
        ])
        param_grid = [
            {
                'scaler': ['passthrough', StandardScaler()], 
                'pca': ['passthrough', PCA(n_components=0.95), PCA(n_components=0.99)], 
                'hgb__learning_rate': [0.05, 0.1], 
                'hgb__max_iter': [100, 200], 
                'hgb__max_depth': [5, 10, 20]
            }
        ]
        grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        clean_params = {}
        for k, v in grid.best_params_.items():
            if k == 'pca' and isinstance(v, PCA): 
                clean_params['pca_status'] = 'active'
                clean_params['pca__n_components'] = v.n_components
            elif k == 'pca' and v == 'passthrough': 
                clean_params['pca_status'] = 'passthrough'
            elif k != 'scaler': 
                clean_params[k] = v
        with open(params_path, 'w') as f: 
            json.dump(clean_params, f, indent=4)
        console.print(f"[bold green]Grid Search completed! Best params saved to {params_path}[/bold green]")
        return best_model
    else:
        console.print("[yellow]Loading parameters from JSON without Grid Search...[/yellow]")
        base_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA()), 
            ('hgb', HistGradientBoostingClassifier(random_state=SEED, early_stopping=False))
        ])
        if os.path.exists(params_path):
            with open(params_path, 'r') as f: loaded_params = json.load(f)
            if loaded_params.get('pca_status') == 'passthrough' or loaded_params.get('pca') == 'passthrough':
                base_pipeline.set_params(pca='passthrough')
            else: 
                base_pipeline.set_params(pca=PCA(n_components=loaded_params.get('pca__n_components', 0.95)))
                
            model_params = {k: v for k, v in loaded_params.items() if k.startswith('hgb__')}
            base_pipeline.set_params(scaler='passthrough', **model_params)
        else:
            console.print("[red]JSON non trovato! Uso i parametri di default.[/red]")
            base_pipeline.set_params(scaler='passthrough', pca='passthrough', hgb__learning_rate=0.1, hgb__max_iter=100)
            
        base_pipeline.fit(X_train, y_train)
        return base_pipeline

def calculate_operational_point(FAR_array, DIR_array, thresholds, target_far):
    idx = np.where(FAR_array <= target_far)[0]
    if len(idx) > 0:
        best_idx = idx[0]
        return thresholds[best_idx], DIR_array[best_idx], FAR_array[best_idx]
    return None, None, None

def run_analysis(valid_subs, incomplete_subs):
    X_train, y_train_str, X_gen, y_gen_str, X_imp = load_data(valid_subs, incomplete_subs)
    
    unique_known = sorted(list(set(y_train_str)))
    label_map = {name: idx for idx, name in enumerate(unique_known)}
    rev_label_map = {idx: name for name, idx in label_map.items()}
    y_train = np.array([label_map[s] for s in y_train_str])
    y_gen = np.array([label_map[s] for s in y_gen_str])
    
    model = build_or_train_pipeline(X_train, y_train)
    
    console.print("Extracting probabilities...")
    probs_gen = model.predict_proba(X_gen)
    probs_imp = model.predict_proba(X_imp)
    
    TG = len(probs_gen)
    TI_virtual = len(probs_gen) 
    TI_true = len(probs_imp)
    TI_total = TI_virtual + TI_true 
    
    thresholds = np.linspace(0.0, 1.0, 500)
    DIR_list, FAR_list, FRR_list = [], [], []
    
    genuine_confidences = []
    impostor_confidences = []
    subject_confidences = {subj: [] for subj in unique_known}
    
    for i in range(TG):
        if np.argmax(probs_gen[i]) == y_gen[i]:
            conf = np.max(probs_gen[i])
            genuine_confidences.append(conf)
            subject_confidences[rev_label_map[y_gen[i]]].append(conf)
            
        masked_probs = probs_gen[i].copy()
        masked_probs[y_gen[i]] = -1.0 
        impostor_confidences.append(np.max(masked_probs))
        
    for i in range(TI_true):
        impostor_confidences.append(np.max(probs_imp[i]))
        
    for t in thresholds:
        DI, FA = 0, 0
        
        for i in range(TG):
            if np.max(probs_gen[i]) >= t and np.argmax(probs_gen[i]) == y_gen[i]:
                DI += 1
                
        for i in range(TI_virtual):
            masked_probs = probs_gen[i].copy()
            masked_probs[y_gen[i]] = -1.0 
            if np.max(masked_probs) >= t:
                FA += 1
                
        for i in range(TI_true):
            if np.max(probs_imp[i]) >= t:
                FA += 1
                
        DIR_list.append(DI / TG if TG > 0 else 0)
        FAR_list.append(FA / TI_total if TI_total > 0 else 0)
        FRR_list.append(1 - (DI / TG) if TG > 0 else 1) 
        
    DIR_array, FAR_array, FRR_array = np.array(DIR_list), np.array(FAR_list), np.array(FRR_list)
    
    def eer_func(t):
        return interp1d(thresholds, FAR_array)(t) - interp1d(thresholds, FRR_array)(t)

    eer_threshold = brentq(eer_func, 0.0, 1.0)
    eer_value = float(interp1d(thresholds, FAR_array)(eer_threshold))
    dir_at_eer = float(interp1d(thresholds, DIR_array)(eer_threshold))
    
    sort_idx = np.argsort(FAR_array)
    roc_auc = auc(FAR_array[sort_idx], DIR_array[sort_idx])
    
    t_1, dir_1, far_1 = calculate_operational_point(FAR_array, DIR_array, thresholds, 0.01)
    t_5, dir_5, far_5 = calculate_operational_point(FAR_array, DIR_array, thresholds, 0.05)
    t_10, dir_10, far_10 = calculate_operational_point(FAR_array, DIR_array, thresholds, 0.10)

    plt.figure(figsize=(9, 6))
    plt.plot(FAR_array, DIR_array, label=f'Watchlist ROC (AUC = {roc_auc:.4f})', color='brown', linewidth=2.5)
    plt.plot(eer_value, dir_at_eer, 'ro', markersize=8, label=f'EER Point ({eer_value*100:.2f}%)')
    plt.title('Masked Open Set ROC - HGB (Walk/Stairs)', fontsize=14, pad=15)
    plt.xlabel('False Alarm Rate (FAR)', fontsize=12)
    plt.ylabel('Detect & Identify Rate (DIR)', fontsize=12)
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_hgb_walk_stairs.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(genuine_confidences, fill=True, color="green", label="Genuine Attempts (Correct ID)", alpha=0.5)
    sns.kdeplot(impostor_confidences, fill=True, color="red", label="Impostor Attempts (Masked + Strangers)", alpha=0.5)
    plt.title('Score Distribution - HGB (Walk/Stairs)', fontsize=14, pad=15)
    plt.xlabel('Confidence Score (Probability)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'dist_hgb_walk_stairs.png'), dpi=300)
    plt.close()
    
    report_path = os.path.join(RESULTS_DIR, "hgb_walk_stairs_comprehensive_report.txt")
    with open(report_path, "w") as f:
        f.write("==================================================================\n")
        f.write(" COMPREHENSIVE OPEN SET REPORT - HGB (WALK/STAIRS)\n")
        f.write("==================================================================\n\n")
        f.write("1. DATASET COMPOSITION:\n")
        f.write(f"   - Registered Subjects (Gallery): {len(valid_subs)}\n")
        f.write(f"   - Unknown Subjects (True Impostors): {len(incomplete_subs)}\n")
        f.write(f"   - Total Genuine Attempts (TG): {TG}\n")
        f.write(f"   - Total Impostor Attempts (TI): {TI_total} ({TI_virtual} Virtual + {TI_true} Strangers)\n\n")
        
        f.write("2. GLOBAL METRICS:\n")
        f.write(f"   - Equal Error Rate (EER): {eer_value*100:.2f}%\n")
        f.write(f"   - EER Balance Threshold: {eer_threshold:.4f}\n")
        f.write(f"   - FAR at EER Point: {eer_value*100:.2f}%\n")
        f.write(f"   - FRR at EER Point: {eer_value*100:.2f}%\n")
        f.write(f"   - Genuine Reject Rate (GRR) at EER: {100 - (eer_value*100):.2f}%\n")
        f.write(f"   - ROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("3. OPERATIONAL POINTS (HIGH SECURITY):\n")
        if t_1 is not None:
            f.write("   Maximum Allowable FAR: 1.0%\n")
            f.write(f"     -> Threshold Required: >= {t_1:.4f}\n")
            f.write(f"     -> Resulting DIR: {dir_1*100:.2f}%\n")
            f.write(f"     -> Actual FAR: {far_1*100:.2f}%\n\n")
        
        if t_5 is not None:
            f.write("   Maximum Allowable FAR: 5.0%\n")
            f.write(f"     -> Threshold Required: >= {t_5:.4f}\n")
            f.write(f"     -> Resulting DIR: {dir_5*100:.2f}%\n")
            f.write(f"     -> Actual FAR: {far_5*100:.2f}%\n\n")
        
        if t_10 is not None:
            f.write("   Maximum Allowable FAR: 10.0%\n")
            f.write(f"     -> Threshold Required: >= {t_10:.4f}\n")
            f.write(f"     -> Resulting DIR: {dir_10*100:.2f}%\n")
            f.write(f"     -> Actual FAR: {far_10*100:.2f}%\n\n")
        
        f.write("4. CONFIDENCE ANALYSIS:\n")
        f.write(f"   - Overall Average Genuine Confidence: {np.mean(genuine_confidences):.4f} (std: {np.std(genuine_confidences):.4f})\n")
        f.write(f"   - Overall Average Impostor Confidence: {np.mean(impostor_confidences):.4f} (std: {np.std(impostor_confidences):.4f})\n\n")
        f.write("   Average Genuine Confidence per Subject (Gallery 1-19):\n")
        for subj, confs in subject_confidences.items():
            if len(confs) > 0:
                f.write(f"     * {subj}: {np.mean(confs):.4f} (based on {len(confs)} correct predictions)\n")
            else:
                f.write(f"     * {subj}: 0.0000 (0 correct predictions)\n")
                
    console.print(f"\n[bold green]Success! Analysis finished.[/bold green]")
    console.print(f"Metrics: EER = {eer_value*100:.2f}%, AUC = {roc_auc:.4f}")
    console.print(f"Results saved in: [cyan]{RESULTS_DIR}[/cyan]")

def main():
    console.print(Panel.fit("[bold magenta]COMPREHENSIVE OPEN SET ANALYSIS - HGB WALK/STAIRS[/bold magenta]"))
    valid_subs, incomplete_subs = get_subject_pools()
    if not valid_subs: 
        console.print("[red]No valid subjects found. Check paths.[/red]")
        return
    run_analysis(valid_subs, incomplete_subs)

if __name__ == "__main__":
    main()
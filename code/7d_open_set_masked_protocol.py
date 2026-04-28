import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
MODEL_DIR = "models"
RESULTS_DIR = "results/masked_open_set"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
            walk_count, stairs_count, slope_count = 0, 0, 0
            for root, dirs, files in os.walk(run_dir):
                root_lower = root.lower()
                if 'debug' in root_lower or "_backup" in root_lower: continue
                for f in files:
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        if 'slope' in root_lower: slope_count += 1
                        elif 'walk' in f.lower(): walk_count += 1
                        elif 'stairs' in f.lower() or 'up' in f.lower() or 'down' in f.lower(): stairs_count += 1
            if walk_count < 6 or stairs_count < 6 or slope_count < 6:
                is_complete = False; break
        if is_complete: valid_subjects.append(subj)

    incomplete_subjects = [s for s in all_subjects if s not in valid_subjects]
    return sorted(valid_subjects), sorted(incomplete_subjects)

def load_data(valid_subs, incomplete_subs, data_key):
    X_train, y_train_str = [], []
    X_gen, y_gen_str = [], []
    X_imp = []
    
    for subj in valid_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            root_lower = root.lower()
            if 'debug' in root_lower or "_backup" in root_lower: continue
            is_slope = "slope" in root_lower
            if (data_key == 'slope' and not is_slope) or (data_key == 'walk_stairs' and is_slope):
                continue
                
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                    if data_key == 'slope' and len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
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
            if 'debug' in root_lower or "_backup" in root_lower: continue
            is_slope = "slope" in root_lower
            if (data_key == 'slope' and not is_slope) or (data_key == 'walk_stairs' and is_slope):
                continue
                
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                    if data_key == 'slope' and len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
                except: continue
                X_imp.append(vec)
                
    return (np.nan_to_num(X_train), np.array(y_train_str), 
            np.nan_to_num(X_gen), np.array(y_gen_str), 
            np.nan_to_num(X_imp))

def build_pipeline(task):
    params_path = os.path.join(MODEL_DIR, f"best_params_{task}_closed_set.json")
    if task == "rf_walk_stairs":
        base_pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('rf', RandomForestClassifier(random_state=SEED, n_jobs=-1))])
    else:
        base_pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('svm', SVC(probability=True, random_state=SEED))])
        
    if os.path.exists(params_path):
        with open(params_path, 'r') as f: loaded_params = json.load(f)
        if loaded_params.get('pca_status') == 'passthrough':
            base_pipeline.set_params(pca='passthrough')
        else: 
            base_pipeline.set_params(pca=PCA(n_components=loaded_params.get('pca__n_components', 0.95)))
            
        model_params = {k: v for k, v in loaded_params.items() if k.startswith('rf__') or k.startswith('svm__')}
        base_pipeline.set_params(scaler='passthrough' if task=="rf_walk_stairs" else StandardScaler(), **model_params)
    return base_pipeline

def calculate_masked_metrics(probs_gen, y_gen, probs_imp, thresholds):
    TG = len(probs_gen)
    TI_virtual = len(probs_gen) 
    TI_true = len(probs_imp)
    TI_total = TI_virtual + TI_true 
    
    DIR_list, FAR_list, FRR_list = [], [], []
    
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
        
    return np.array(DIR_list), np.array(FAR_list), np.array(FRR_list), TG, TI_virtual, TI_true, TI_total

def run_masked_protocol(valid_subs, incomplete_subs, task_name, data_key):
    console.print(f"\n[bold cyan]--- Running Virtual Impostor Protocol for {task_name} ---[/bold cyan]")
    
    X_train, y_train_str, X_gen, y_gen_str, X_imp = load_data(valid_subs, incomplete_subs, data_key)
    
    unique_known = sorted(list(set(y_train_str)))
    label_map = {name: idx for idx, name in enumerate(unique_known)}
    y_train = np.array([label_map[s] for s in y_train_str])
    y_gen = np.array([label_map[s] for s in y_gen_str])
    
    console.print(f"Training model on all {len(valid_subs)} valid subjects...")
    model = build_pipeline(task_name)
    model.fit(X_train, y_train)
    
    console.print("Extracting probabilities...")
    probs_gen = model.predict_proba(X_gen)
    probs_imp = model.predict_proba(X_imp)
    
    thresholds = np.linspace(0.0, 1.0, 200)
    DIR, FAR, FRR, TG, TI_virt, TI_true, TI_tot = calculate_masked_metrics(probs_gen, y_gen, probs_imp, thresholds)
    
    def eer_func(t):
        return interp1d(thresholds, FAR)(t) - interp1d(thresholds, FRR)(t)

    eer_threshold = brentq(eer_func, 0.0, 1.0)
    eer_value = float(interp1d(thresholds, FAR)(eer_threshold))
    dir_at_eer = float(interp1d(thresholds, DIR)(eer_threshold))
    
    sort_idx = np.argsort(FAR)
    roc_auc = auc(FAR[sort_idx], DIR[sort_idx])
    
    plt.figure(figsize=(10, 6))
    plt.plot(FAR, DIR, label=f'Watchlist ROC (AUC = {roc_auc:.4f})', color='#1f77b4' if 'rf' in task_name else '#2ca02c', linewidth=2)
    plt.plot(eer_value, dir_at_eer, 'ro', markersize=8, label=f'EER Point ({eer_value*100:.2f}%)')
    plt.title(f'Masked Open Set ROC - {"Random Forest" if "rf" in task_name else "Linear SVM"}', fontsize=14)
    plt.xlabel('False Alarm Rate (FAR)', fontsize=12)
    plt.ylabel('Detect & Identify Rate (DIR)', fontsize=12)
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'roc_masked_{task_name}.png'), dpi=300)
    plt.close()
    
    console.print(f"[bold green]Done! Masked Protocol EER for {task_name}: {eer_value*100:.2f}%[/bold green]")
    
    return eer_value*100, eer_threshold, roc_auc, TG, TI_virt, TI_true, TI_tot

def main():
    console.print(Panel.fit("[bold magenta]VIRTUAL IMPOSTOR MASKING PROTOCOL[/bold magenta]"))
    
    valid_subs, incomplete_subs = get_subject_pools()
    if not valid_subs: return
    
    eer_rf, t_rf, auc_rf, tg_rf, tv_rf, tt_rf, tot_i_rf = run_masked_protocol(valid_subs, incomplete_subs, "rf_walk_stairs", "walk_stairs")
    eer_svm, t_svm, auc_svm, tg_svm, tv_svm, tt_svm, tot_i_svm = run_masked_protocol(valid_subs, incomplete_subs, "svm_slope", "slope")
    
    report_path = os.path.join(RESULTS_DIR, "masked_protocol_detailed_report.txt")
    with open(report_path, "w") as f:
        f.write("=================================================\n")
        f.write(" VIRTUAL IMPOSTOR MASKING PROTOCOL -  REPORT\n")
        f.write("=================================================\n")
        f.write("- Registered Subjects (Gallery): 19\n")
        f.write("- Unknown Subjects (True Impostors): 7\n\n")
        
        f.write("--- RANDOM FOREST (Walk/Stairs) ---\n")
        f.write(f"Total Genuine Attempts (TG): {tg_rf} (Used for DIR and FRR)\n")
        f.write(f"Total Impostor Attempts (TI): {tot_i_rf} (Used for FAR)\n")
        f.write(f"   -> Virtual Impostors (Gallery masked): {tv_rf}\n")
        f.write(f"   -> True Impostors (Strangers): {tt_rf}\n")
        f.write(f"RESULTS:\n")
        f.write(f"   - Equal Error Rate (EER): {eer_rf:.2f}%\n")
        f.write(f"   - EER Balance Threshold: {t_rf:.4f}\n")
        f.write(f"   - Genuine Reject Rate (GRR): {100 - eer_rf:.2f}%\n")
        f.write(f"   - ROC AUC: {auc_rf:.4f}\n\n")
        
        f.write("-" * 50 + "\n\n")
        
        f.write("--- LINEAR SVM (Slope) ---\n")
        f.write(f"Total Genuine Attempts (TG): {tg_svm} (Used for DIR and FRR)\n")
        f.write(f"Total Impostor Attempts (TI): {tot_i_svm} (Used for FAR)\n")
        f.write(f"   -> Virtual Impostors (Gallery masked): {tv_svm}\n")
        f.write(f"   -> True Impostors (Strangers): {tt_svm}\n")
        f.write(f"RESULTS:\n")
        f.write(f"   - Equal Error Rate (EER): {eer_svm:.2f}%\n")
        f.write(f"   - EER Balance Threshold: {t_svm:.4f}\n")
        f.write(f"   - Genuine Reject Rate (GRR): {100 - eer_svm:.2f}%\n")
        f.write(f"   - ROC AUC: {auc_svm:.4f}\n")
        
    table = Table(title="Masked Protocol Results", box=box.ROUNDED)
    table.add_column("Model & Task", style="cyan")
    table.add_column("EER (%)", justify="right", style="green")
    table.add_column("EER Threshold", justify="right", style="yellow")
    table.add_column("AUC", justify="right", style="magenta")
    table.add_row("Random Forest (Walk/Stairs)", f"{eer_rf:.2f}%", f"{t_rf:.2f}", f"{auc_rf:.4f}")
    table.add_row("Linear SVM (Slope)", f"{eer_svm:.2f}%", f"{t_svm:.2f}", f"{auc_svm:.4f}")
    console.print(table)
    console.print(f"\n[bold green]Detailed report saved to: {report_path}[/bold green]")
    console.print(f"[bold green]ROC Curves saved to: {RESULTS_DIR}[/bold green]")

if __name__ == "__main__":
    main()
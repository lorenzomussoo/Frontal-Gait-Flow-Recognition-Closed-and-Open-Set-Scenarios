import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from joblib import load
import warnings
from rich.console import Console
from rich.panel import Panel

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models"

VIDEO_CUT_INDEX = 73728
NUM_KNOWN_SUBJECTS = 15
SEED = 42
USE_ALL_VALID_AS_KNOWN = False

def get_subject_pools(is_slope):
    np.random.seed(SEED)
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
            w_c, st_c, sl_c = 0, 0, 0
            for root, dirs, files in os.walk(run_dir):
                dirs.sort()
                root_lower = root.lower()
                if 'debug' in root_lower or '_backup' in root_lower: continue
                if is_slope and "slope" not in root_lower: continue
                if not is_slope and "slope" in root_lower: continue
                
                for f in sorted(files): 
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        if is_slope:
                            sl_c += 1
                        else:
                            f_l = f.lower()
                            if 'walk' in f_l: w_c += 1
                            elif 'stairs' in f_l or 'up' in f_l or 'down' in f_l: st_c += 1
                            
            if is_slope:
                if sl_c < 6: is_complete = False; break
            else:
                if w_c < 6 or st_c < 6: is_complete = False; break
                
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

def load_data(known_subs, unknown_subs, is_slope):
    X_gen, y_gen_str, X_imp = [], [], []
    
    for subj in known_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            dirs.sort()
            root_lower = root.lower()
            if 'ThirdRun' in root and 'debug' not in root_lower and '_backup' not in root_lower:
                if is_slope and "slope" not in root_lower: continue
                if not is_slope and "slope" in root_lower: continue
                for f in sorted(files):
                    if f.endswith('.npy') and not f.startswith('._') and 'flip' not in f:
                        try:
                            vec = np.load(os.path.join(root, f))
                            if is_slope:
                                if len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
                                else: continue
                            X_gen.append(vec)
                            y_gen_str.append(subj)
                        except: pass

    for subj in unknown_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            dirs.sort()
            root_lower = root.lower()
            if 'debug' not in root_lower and '_backup' not in root_lower:
                if is_slope and "slope" not in root_lower: continue
                if not is_slope and "slope" in root_lower: continue
                for f in sorted(files):
                    if f.endswith('.npy') and not f.startswith('._') and 'flip' not in f:
                        try:
                            vec = np.load(os.path.join(root, f))
                            if is_slope:
                                if len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
                                else: continue
                            X_imp.append(vec)
                        except: pass
                        
    return np.nan_to_num(X_gen), np.array(y_gen_str), np.nan_to_num(X_imp)

def plot_distributions(model_name, is_slope, title):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        console.print(f"[red]Model {model_name} not found! Run the Open Set script first.[/red]")
        return
        
    known, unknown = get_subject_pools(is_slope)
    X_gen, y_gen_str, X_imp = load_data(known, unknown, is_slope)
    
    le = LabelEncoder()
    le.fit(known)
    y_gen = le.transform(y_gen_str)
    
    model = load(model_path)
    probs_gen = model.predict_proba(X_gen)
    probs_imp = model.predict_proba(X_imp)
    
    genuine_scores = []
    for i in range(len(X_gen)):
        if np.argmax(probs_gen[i]) == y_gen[i]:
            genuine_scores.append(np.max(probs_gen[i]))
            
    impostor_scores = np.max(probs_imp, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(genuine_scores, fill=True, color="green", label="Genuine Attempts (Correct ID)", alpha=0.5)
    sns.kdeplot(impostor_scores, fill=True, color="red", label="Impostor Attempts (False Alarms)", alpha=0.5)
    
    mode_str = "19 Knowns" if USE_ALL_VALID_AS_KNOWN else "15 Knowns"
    plt.title(f'Score Distribution ({mode_str}) - {title}')
    plt.xlabel('Confidence Score (Probability)')
    plt.ylabel('Density')
    plt.xlim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    suffix = "_19_known" if USE_ALL_VALID_AS_KNOWN else "_15_known"
    save_name = f"dist_{'slope' if is_slope else 'walk_stairs'}{suffix}.png"
    RESULTS_DIR = "results/rf_walk_stairs/open_set" if not is_slope else "results/svm_slope/open_set"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, save_name), dpi=300)
    plt.close()
    console.print(f"[green]Saved Distribution for {title} -> {save_name}[/green]")

def main():
    mode = "19 Subjects" if USE_ALL_VALID_AS_KNOWN else "15 Subjects"
    console.print(Panel.fit(f"[bold cyan]Generating Score Distributions for Open Set ({mode})...[/bold cyan]"))
    
    if USE_ALL_VALID_AS_KNOWN:
        rf_model = "rf_walk_stairs_closed_set.joblib" 
        svm_model = "svm_slope_closed_set.joblib"
    else:
        rf_model = "rf_walk_stairs_open_set.joblib"
        svm_model = "svm_slope_open_set.joblib"
        
    plot_distributions(rf_model, False, "Random Forest (Walk/Stairs)")
    plot_distributions(svm_model, True, "SVM (Slope)")

if __name__ == "__main__":
    main()
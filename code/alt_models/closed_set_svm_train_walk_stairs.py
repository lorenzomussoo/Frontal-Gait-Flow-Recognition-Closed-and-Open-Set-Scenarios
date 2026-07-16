import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import box

np.random.seed(42)
warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models/alt_models"
RESULTS_DIR = "results/alt_models/svm_walk_stairs/no_flip/closed_set"
MODEL_NAME = "svm_walk_stairs_closed_set_no_flip.joblib"
PARAMS_FILE = os.path.join(MODEL_DIR, 'best_params_svm_walk_stairs_closed_set_no_flip.json')

PERFORM_GRID_SEARCH = False
SAVE_RESULTS = False
SEED = 42

if SAVE_RESULTS:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    console.print(Panel.fit(f"[bold yellow]Loading Walk/Stairs Dataset for SVM: {FEATURES_ROOT}[/bold yellow]"))
    X_train, y_train_str, X_test, y_test_str, meta_test = [], [], [], [], []
    if not os.path.exists(FEATURES_ROOT): return None

    all_subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    candidate_subjects = [s for s in all_subjects if os.path.exists(os.path.join(FEATURES_ROOT, s, "FirstRun")) and 
                                                     os.path.exists(os.path.join(FEATURES_ROOT, s, "SecondRun")) and
                                                     os.path.exists(os.path.join(FEATURES_ROOT, s, "ThirdRun"))]
            
    valid_subjects = []
    for subj in candidate_subjects:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        is_complete = True
        for run_name in ["FirstRun", "SecondRun", "ThirdRun"]:
            run_dir = os.path.join(subj_path, run_name)
            walk_count, stairs_count = 0, 0
            for root, dirs, files in os.walk(run_dir):
                dirs.sort()
                root_lower = root.lower()
                if 'debug' in root_lower or "slope" in root_lower or "_backup" in root_lower: continue
                for f in sorted(files):
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        f_lower = f.lower()
                        if 'walk' in f_lower: walk_count += 1
                        elif 'stairs' in f_lower or 'up' in f_lower or 'down' in f_lower: stairs_count += 1
            if walk_count < 6 or stairs_count < 6: is_complete = False; break
        if is_complete: valid_subjects.append(subj)

    for subj in track(valid_subjects, description="Loading valid subjects..."): 
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            dirs.sort()
            root_lower = root.lower()
            if 'debug' in root_lower or "slope" in root_lower or "_backup" in root_lower: continue
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                file_path = os.path.join(root, f)
                try: vector = np.load(file_path)
                except: continue
                
                if 'FirstRun' in file_path or 'SecondRun' in file_path:
                    X_train.append(vector); y_train_str.append(subj)
                elif 'ThirdRun' in file_path:
                    X_test.append(vector); y_test_str.append(subj); meta_test.append(f)
                    
    return np.array(X_train), np.array(X_test), np.array(y_train_str), np.array(y_test_str), meta_test

def train_and_evaluate():
    data = load_data()
    if data is None: return
    X_train, X_test, y_train_str, y_test_str, meta_test = data

    if np.isnan(X_train).any(): X_train, X_test = np.nan_to_num(X_train), np.nan_to_num(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_test = label_encoder.transform(y_test_str)
    classes = label_encoder.classes_

    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()), 
        ('svm', SVC(probability=True, random_state=SEED))
    ])

    if PERFORM_GRID_SEARCH:
        param_grid = [
            {'scaler': [StandardScaler()], 'pca': ['passthrough', PCA(n_components=0.95)], 
             'svm__C': [0.1, 1.0, 10.0], 'svm__kernel': ['linear', 'rbf']}
        ]
        search = GridSearchCV(base_pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
        search.fit(X_train, y_train)
        final_model = search.best_estimator_
        
        if SAVE_RESULTS:
            best_params_ser = {}
            for k, v in search.best_params_.items():
                if k == 'pca' and isinstance(v, PCA): best_params_ser['pca_status'] = 'active'; best_params_ser['pca__n_components'] = v.n_components
                elif k == 'pca' and v == 'passthrough': best_params_ser['pca_status'] = 'passthrough'
                elif k != 'scaler': best_params_ser[k] = v
            with open(PARAMS_FILE, 'w') as f: json.dump(best_params_ser, f, indent=4)
            dump(search, os.path.join(MODEL_DIR, MODEL_NAME))
    else:
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, 'r') as f: loaded_params = json.load(f)
            if loaded_params.get('pca_status') == 'passthrough': base_pipeline.set_params(pca='passthrough')
            else: base_pipeline.set_params(pca=PCA(n_components=loaded_params.get('pca__n_components', 0.95)))
            svm_params = {k: v for k, v in loaded_params.items() if k.startswith('svm__')}
            base_pipeline.set_params(scaler=StandardScaler(), **svm_params)
        else:
            base_pipeline.set_params(scaler=StandardScaler(), pca=PCA(n_components=0.95), svm__C=1.0, svm__kernel='linear')
        base_pipeline.fit(X_train, y_train)
        final_model = base_pipeline

    console.print("[bold yellow]Evaluating Model...[/bold yellow]")
    y_pred = final_model.predict(X_test)
    y_pred_str_out = label_encoder.inverse_transform(y_pred)
    probs = final_model.predict_proba(X_test)

    report_text = classification_report(y_test_str, y_pred_str_out)
    console.print(report_text)

    TA, G = len(y_test), len(classes)
    ranks_counts = np.zeros(G)
    for i in range(TA):
        sorted_indices = np.argsort(probs[i])[::-1]
        ranks_counts[np.where(sorted_indices == y_test[i])[0][0]] += 1
    cms = np.cumsum(ranks_counts) / TA * 100

    cmc_table = Table(title="Closed Set Metrics (CMC)", box=box.ROUNDED)
    cmc_table.add_column("Rank", justify="center", style="magenta"); cmc_table.add_column("Cumulative Match Score", justify="center", style="green")
    for k in [1, 2, 3, 4, 5, 10]:
        if k <= G: cmc_table.add_row(f"Rank-{k}", f"{cms[k-1]:.2f}%")
    console.print(cmc_table)

    if SAVE_RESULTS:
        with open(os.path.join(RESULTS_DIR, "metrics_report.txt"), "w") as f:
            f.write("=== FINAL REPORT: SVM (WALK/STAIRS) ===\n\n--- CLASSIFICATION REPORT ---\n")
            f.write(report_text + "\n\n--- CUMULATIVE MATCH SCORE (CMS) ---\n")
            for k in [1, 2, 3, 4, 5, 10]:
                if k <= G: f.write(f"Rank-{k}: {cms[k-1]:.2f}%\n")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, G + 1), cms, marker='o', linestyle='-', color='indigo', linewidth=2)
        plt.title('Cumulative Match Characteristic (CMC) - SVM Walk/Stairs')
        plt.xlabel('Rank'); plt.ylabel('Recognition Rate (%)'); plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(np.arange(1, min(G + 1, 21), 1)); plt.ylim(0, 105); plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'cmc_curve.png'), dpi=300); plt.close()

        cm = confusion_matrix(y_test_str, y_pred_str_out, labels=classes)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - SVM Walk/Stairs')
        plt.ylabel('True Identity (Probe)'); plt.xlabel('Predicted Identity (Gallery Match)')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300); plt.close()

if __name__ == "__main__":
    train_and_evaluate()
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import load
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
MODEL_DIR = "models"
SEED = 42

L = 24576
IDX_DEPTH = (0, L)
IDX_RGB = (L, 2*L)
IDX_IR = (2*L, 3*L)
IDX_IMU_START = 3*L

def load_dataset(is_slope=False):
    dataset_name = "SLOPE" if is_slope else "WALK/STAIRS"
    console.print(Panel.fit(f"[bold cyan]Loading {dataset_name} Dataset for Analysis...[/bold cyan]"))
    
    X_train, y_train_str, X_test, y_test_str = [], [], [], []
    
    if not os.path.exists(FEATURES_ROOT):
        console.print("[red]Error: Features folder not found![/red]")
        return None

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
            walk_count, stairs_count, slope_count = 0, 0, 0
            
            for root, _, files in os.walk(run_dir):
                if 'debug' in root or "_backup" in root.lower(): continue
                if is_slope and "slope" not in root.lower(): continue
                if not is_slope and "slope" in root.lower(): continue
                
                for f in files:
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        if is_slope: slope_count += 1
                        else:
                            f_lower = f.lower()
                            if 'walk' in f_lower: walk_count += 1
                            elif 'stairs' in f_lower or 'up' in f_lower or 'down' in f_lower: stairs_count += 1
            
            if is_slope:
                if slope_count < 6: is_complete = False; break
            else:
                if walk_count < 6 or stairs_count < 6: is_complete = False; break
                
        if is_complete: valid_subjects.append(subj)

    for subj in track(valid_subjects, description=f"Loading {dataset_name} subjects..."): 
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            dirs.sort()
            if 'debug' in root or "_backup" in root.lower(): continue
            if is_slope and "slope" not in root.lower(): continue
            if not is_slope and "slope" in root.lower(): continue
                
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._'): continue
                file_path = os.path.join(root, f)
                try: 
                    vector = np.load(file_path)
                    if is_slope:
                        if len(vector) > IDX_IMU_START: vector = vector[IDX_IMU_START:]
                        else: continue
                except: continue
                
                if 'FirstRun' in file_path or 'SecondRun' in file_path:
                    X_train.append(vector)
                    y_train_str.append(subj)
                elif 'ThirdRun' in file_path:
                    X_test.append(vector)
                    y_test_str.append(subj)
                    
    X_train = np.nan_to_num(np.array(X_train))
    X_test = np.nan_to_num(np.array(X_test))
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_test = label_encoder.transform(y_test_str)
    
    return X_train, X_test, y_train, y_test, label_encoder.classes_

def build_rf_pipeline():
    params_file = os.path.join(MODEL_DIR, 'best_params_rf_walk_stairs_closed_set.json')
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()), 
        ('rf', RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])
    
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            loaded_params = json.load(f)
        
        is_passthrough = (loaded_params.get('pca_status') == 'passthrough') or (loaded_params.get('pca') == 'passthrough')
        if is_passthrough: base_pipeline.set_params(pca='passthrough')
        else: base_pipeline.set_params(pca=PCA(n_components=loaded_params.get('pca__n_components', 0.95)))
            
        rf_params = {k: v for k, v in loaded_params.items() if k.startswith('rf__')}
        base_pipeline.set_params(scaler='passthrough', **rf_params)
    else:
        base_pipeline.set_params(scaler='passthrough', pca='passthrough', rf__n_estimators=500, rf__min_samples_split=2)
        
    return base_pipeline

def perform_ablation(X_train, X_test, y_train, y_test, results_dir):
    console.print("[bold yellow]Performing Ablation Study (15 Combinations)...[/bold yellow]")
    
    modalities = {
        'Fusion All (Depth+RGB+IR+IMU)': lambda x: x,

        'Depth': lambda x: x[:, IDX_DEPTH[0]:IDX_DEPTH[1]],
        'RGB': lambda x: x[:, IDX_RGB[0]:IDX_RGB[1]],
        'IR': lambda x: x[:, IDX_IR[0]:IDX_IR[1]],
        'IMU': lambda x: x[:, IDX_IMU_START:],
        
        'Depth+RGB': lambda x: x[:, :IDX_RGB[1]],
        'Depth+IR': lambda x: np.concatenate([x[:, IDX_DEPTH[0]:IDX_DEPTH[1]], x[:, IDX_IR[0]:IDX_IR[1]]], axis=1),
        'Depth+IMU': lambda x: np.concatenate([x[:, IDX_DEPTH[0]:IDX_DEPTH[1]], x[:, IDX_IMU_START:]], axis=1),
        'RGB+IR': lambda x: x[:, IDX_RGB[0]:IDX_IR[1]],
        'RGB+IMU': lambda x: np.concatenate([x[:, IDX_RGB[0]:IDX_RGB[1]], x[:, IDX_IMU_START:]], axis=1),
        'IR+IMU': lambda x: x[:, IDX_IR[0]:],
        
        'Depth+RGB+IR': lambda x: x[:, :IDX_IR[1]],
        'Depth+RGB+IMU': lambda x: np.concatenate([x[:, :IDX_RGB[1]], x[:, IDX_IMU_START:]], axis=1),
        'Depth+IR+IMU': lambda x: np.concatenate([x[:, IDX_DEPTH[0]:IDX_DEPTH[1]], x[:, IDX_IR[0]:]], axis=1),
        'RGB+IR+IMU': lambda x: x[:, IDX_RGB[0]:]
    }

    results = []
    G = len(np.unique(y_test))
    
    for mod_name, slice_func in track(modalities.items(), description="Training models..."):
        X_train_slice = slice_func(X_train)
        X_test_slice = slice_func(X_test)

        np.random.seed(SEED)
        
        model = build_rf_pipeline()
        model.fit(X_train_slice, y_train)
        y_pred = model.predict(X_test_slice)
        probs = model.predict_proba(X_test_slice)
        
        TA = len(y_test)
        ranks_counts = np.zeros(G)
        for i in range(TA):
            sorted_indices = np.argsort(probs[i])[::-1]
            rank = np.where(sorted_indices == y_test[i])[0][0]
            ranks_counts[rank] += 1
            
        cms = np.cumsum(ranks_counts) / TA * 100
        rank1_acc = cms[0]
        f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        dim = X_train_slice.shape[1]
        
        results.append((mod_name, dim, rank1_acc, f1, prec, rec))

    table = Table(title="Ablation Study Results", box=box.DOUBLE_EDGE)
    table.add_column("Modality Combination", style="cyan")
    table.add_column("Dimensions", justify="right")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Macro F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")

    for r in results:
        table.add_row(r[0], str(r[1]), f"{r[2]:.2f}%", f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.4f}")
    console.print(table)

    with open(os.path.join(results_dir, "ablation_study_report.txt"), "w") as f:
        f.write("========================================================================\n")
        f.write("          FULL ABLATION STUDY REPORT (WALK/STAIRS) - RF                 \n")
        f.write("========================================================================\n\n")
        f.write(f"{'Modality':<30} | {'Dimensions':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}\n")
        f.write("-" * 90 + "\n")
        
        for r in results:
            f.write(f"{r[0]:<30} | {r[1]:<10} | {r[2]:>8.2f}% | {r[3]:>8.4f} | {r[4]:>9.4f} | {r[5]:>8.4f}\n")

def perform_confidence_analysis(model, X_test, y_test, classes, results_dir, model_name):
    console.print(f"[bold magenta]Performing Confidence Analysis for {model_name}...[/bold magenta]")
    
    probs = model.predict_proba(X_test)
    confidences = np.max(probs, axis=1)
    predictions_indices = np.argmax(probs, axis=1)
    
    correct_mask = (predictions_indices == y_test)
    incorrect_mask = ~correct_mask
    
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[incorrect_mask]
    
    stats = {
        "Overall Mean": np.mean(confidences) if len(confidences) > 0 else 0,
        "Correct Mean": np.mean(correct_conf) if len(correct_conf) > 0 else 0,
        "Incorrect Mean": np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0,
        "Correct Std": np.std(correct_conf) if len(correct_conf) > 0 else 0,
        "Incorrect Std": np.std(incorrect_conf) if len(incorrect_conf) > 0 else 0
    }

    table = Table(title=f"Confidence Statistics - {model_name}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for k, v in stats.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    with open(os.path.join(results_dir, "confidence_report.txt"), "w") as f:
        f.write(f"=== CONFIDENCE ANALYSIS: {model_name} ===\n\n")
        for k, v in stats.items():
            f.write(f"{k}: {v:.4f}\n")

    plt.figure(figsize=(10, 6))
    if len(correct_conf) > 0:
        sns.histplot(correct_conf, bins=20, color='green', alpha=0.6, label='Correct Predictions', kde=True, stat="density")
    if len(incorrect_conf) > 0:
        sns.histplot(incorrect_conf, bins=20, color='red', alpha=0.6, label='Incorrect Predictions', kde=True, stat="density")
    
    plt.title(f'Prediction Confidence Distribution - {model_name}')
    plt.xlabel('Confidence (Probability of Predicted Class)')
    plt.ylabel('Density')
    plt.xlim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confidence_histogram.png'), dpi=300)
    plt.close()

def main():
    rf_model_path = os.path.join(MODEL_DIR, "rf_walk_stairs_closed_set.joblib")
    svm_model_path = os.path.join(MODEL_DIR, "svm_slope_closed_set.joblib")
    rf_results_dir = "results/rf_walk_stairs/closed_set"
    svm_results_dir = "results/svm_slope/closed_set"

    os.makedirs(rf_results_dir, exist_ok=True)
    os.makedirs(svm_results_dir, exist_ok=True)

    console.print(Panel.fit("[bold green]Analyzing Random Forest (Walk/Stairs)[/bold green]"))
    data_ws = load_dataset(is_slope=False)
    if data_ws:
        X_train_ws, X_test_ws, y_train_ws, y_test_ws, classes_ws = data_ws
        
        perform_ablation(X_train_ws, X_test_ws, y_train_ws, y_test_ws, rf_results_dir)
        
        if os.path.exists(rf_model_path):
            rf_model = load(rf_model_path)
            perform_confidence_analysis(rf_model, X_test_ws, y_test_ws, classes_ws, rf_results_dir, "Random Forest")
        else:
            console.print(f"[red]Model {rf_model_path} not found. Skipping Confidence Analysis for RF.[/red]")

    console.print(Panel.fit("[bold green]Analyzing SVM (Slope)[/bold green]"))
    data_slope = load_dataset(is_slope=True)
    if data_slope:
        _, X_test_slope, _, y_test_slope, classes_slope = data_slope
        
        if os.path.exists(svm_model_path):
            svm_model = load(svm_model_path)
            perform_confidence_analysis(svm_model, X_test_slope, y_test_slope, classes_slope, svm_results_dir, "Support Vector Machine")
        else:
            console.print(f"[red]Model {svm_model_path} not found. Skipping SVM analysis.[/red]")

    console.print("[bold green]Post-hoc analysis complete! Check the results folders.[/bold green]")

if __name__ == "__main__":
    main()
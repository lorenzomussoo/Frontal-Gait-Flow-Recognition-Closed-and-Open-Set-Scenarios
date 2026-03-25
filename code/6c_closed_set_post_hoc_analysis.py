import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from joblib import load
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models"
VIDEO_CUT_INDEX = 49152
SEED = 42

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
                if 'debug' in root: continue
                if is_slope and "slope" not in root.lower(): continue
                if not is_slope and "slope" in root.lower(): continue
                
                for f in files:
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        if is_slope:
                            slope_count += 1
                        else:
                            f_lower = f.lower()
                            if 'walk' in f_lower: walk_count += 1
                            elif 'stairs' in f_lower or 'up' in f_lower or 'down' in f_lower: stairs_count += 1
            
            if is_slope:
                if slope_count < 6: is_complete = False; break
            else:
                if walk_count < 6 or stairs_count < 6: is_complete = False; break
                
        if is_complete:
            valid_subjects.append(subj)

    for subj in track(valid_subjects, description=f"Loading {dataset_name} subjects..."): 
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, _, files in os.walk(subj_path):
            if 'debug' in root: continue
            if is_slope and "slope" not in root.lower(): continue
            if not is_slope and "slope" in root.lower(): continue
                
            for f in files:
                if not f.endswith('.npy') or f.startswith('._'): continue
                file_path = os.path.join(root, f)
                try: 
                    vector = np.load(file_path)
                    if is_slope:
                        if len(vector) > VIDEO_CUT_INDEX: vector = vector[VIDEO_CUT_INDEX:]
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
        if is_passthrough:
            base_pipeline.set_params(pca='passthrough')
        else:
            n_comp = loaded_params.get('pca__n_components', 0.95)
            base_pipeline.set_params(pca=PCA(n_components=n_comp))
            
        rf_params = {k: v for k, v in loaded_params.items() if k.startswith('rf__')}
        base_pipeline.set_params(scaler='passthrough', **rf_params)
    else:
        base_pipeline.set_params(scaler='passthrough', pca='passthrough', rf__n_estimators=500)
        
    return base_pipeline

def perform_ablation(X_train, X_test, y_train, y_test, results_dir):
    console.print("[bold yellow]Performing Ablation Study...[/bold yellow]")
    
    console.print("Training Baseline (Video + IMU)...")
    model_base = build_rf_pipeline()
    model_base.fit(X_train, y_train)
    acc_base = accuracy_score(y_test, model_base.predict(X_test)) * 100

    console.print("Training Video Only...")
    X_train_video = X_train[:, :VIDEO_CUT_INDEX]
    X_test_video = X_test[:, :VIDEO_CUT_INDEX]
    model_video = build_rf_pipeline()
    model_video.fit(X_train_video, y_train)
    acc_video = accuracy_score(y_test, model_video.predict(X_test_video)) * 100

    console.print("Training IMU Only...")
    X_train_imu = X_train[:, VIDEO_CUT_INDEX:]
    X_test_imu = X_test[:, VIDEO_CUT_INDEX:]
    model_imu = build_rf_pipeline()
    model_imu.fit(X_train_imu, y_train)
    acc_imu = accuracy_score(y_test, model_imu.predict(X_test_imu)) * 100

    table = Table(title="Ablation Study Results", box=box.ROUNDED)
    table.add_column("Modality", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Difference vs Baseline", justify="right", style="red")

    table.add_row("Baseline (Video + IMU)", f"{acc_base:.2f}%", "-")
    table.add_row("Video Only (0 - 49152)", f"{acc_video:.2f}%", f"{acc_video - acc_base:.2f}%")
    table.add_row("IMU Only (49152 - end)", f"{acc_imu:.2f}%", f"{acc_imu - acc_base:.2f}%")
    console.print(table)

    with open(os.path.join(results_dir, "ablation_report.txt"), "w") as f:
        f.write("=== ABLATION STUDY REPORT ===\n\n")
        f.write(f"Baseline (Video + IMU) Accuracy: {acc_base:.2f}%\n")
        f.write(f"Video Only Accuracy: {acc_video:.2f}% (Difference: {acc_video - acc_base:.2f}%)\n")
        f.write(f"IMU Only Accuracy: {acc_imu:.2f}% (Difference: {acc_imu - acc_base:.2f}%)\n")

def perform_confidence_analysis(model, X_test, y_test, classes, results_dir, model_name):
    console.print(f"[bold magenta]Performing Confidence Analysis for {model_name}...[/bold magenta]")
    
    probs = model.predict_proba(X_test)
    confidences = np.max(probs, axis=1)
    
    predictions_indices = np.argmax(probs, axis=1)
    
    if len(classes) == probs.shape[1]:
        correct_mask = (predictions_indices == y_test)
    else:
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
import os
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import box

console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
RESULTS_DIR = "results/CCA"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

L = 24576
IDX_VIDEO_END = 3 * L  
IDX_IMU_START = 3 * L  

def load_data_cross_session_no_flips():
    console.print(Panel.fit("[bold cyan]Loading Dataset for PCA-CCA (CROSS-SESSION & NO FLIPS)[/bold cyan]"))
    
    X_vid_train, X_imu_train = [], []
    X_vid_test, X_imu_test = [], []
    
    if not os.path.exists(FEATURES_ROOT):
        console.print("[red]Error: Features folder not found![/red]")
        return None, None, None, None

    all_subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    for subj in track(all_subjects, description="Loading pristine subjects (Cross-Session)..."): 
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            if 'debug' in root or "_backup" in root.lower() or "slope" in root.lower(): continue
            for f in files:
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                
                file_path = os.path.join(root, f)
                try: 
                    vector = np.load(file_path)
                    if np.isnan(vector).any(): continue
                    
                    if 'FirstRun' in file_path or 'SecondRun' in file_path:
                        X_vid_train.append(vector[:IDX_VIDEO_END])
                        X_imu_train.append(vector[IDX_IMU_START:])
                    elif 'ThirdRun' in file_path:
                        X_vid_test.append(vector[:IDX_VIDEO_END])
                        X_imu_test.append(vector[IDX_IMU_START:])
                except: continue
                    
    X_vid_train = np.array(X_vid_train)
    X_imu_train = np.array(X_imu_train)
    X_vid_test = np.array(X_vid_test)
    X_imu_test = np.array(X_imu_test)
    
    console.print(f"\n[bold green]Cross-Session Load Complete (Zero Flips).[/bold green]")
    console.print(f"Train (Run 1 & 2) - Video: {X_vid_train.shape}, IMU: {X_imu_train.shape}")
    console.print(f"Test (Run 3)      - Video: {X_vid_test.shape}, IMU: {X_imu_test.shape}\n")
    
    return X_vid_train, X_imu_train, X_vid_test, X_imu_test

def run_pca_cca_sensitivity():
    X_vid_train, X_imu_train, X_vid_test, X_imu_test = load_data_cross_session_no_flips()
    if X_vid_train is None or len(X_vid_train) == 0: return
    
    console.print("[yellow]1. Standardizing features (fitted on Train only)...[/yellow]")
    scaler_vid = StandardScaler()
    scaler_imu = StandardScaler()
    
    X_vid_train_scaled = scaler_vid.fit_transform(X_vid_train)
    X_vid_test_scaled = scaler_vid.transform(X_vid_test)
    
    X_imu_train_scaled = scaler_imu.fit_transform(X_imu_train)
    X_imu_test_scaled = scaler_imu.transform(X_imu_test)
    
    console.print("[yellow]2. Performing Cross-Session PCA-CCA...[/yellow]")
    
    test_components = [10, 20, 30, 50, 75, 100]
    results = []
    
    table = Table(title="PCA-CCA Sensitivity Results (Cross-Session Test)", box=box.DOUBLE_EDGE)
    table.add_column("PCA Components", justify="center", style="cyan")
    table.add_column("Video Retained Var.", justify="right")
    table.add_column("IMU Retained Var.", justify="right")
    table.add_column("Test Canonical Correlation", justify="right", style="magenta bold")
    
    for k in test_components:
        pca_vid = PCA(n_components=k, random_state=SEED)
        pca_imu = PCA(n_components=k, random_state=SEED)
        
        X_vid_train_pca = pca_vid.fit_transform(X_vid_train_scaled)
        X_imu_train_pca = pca_imu.fit_transform(X_imu_train_scaled)
        
        X_vid_test_pca = pca_vid.transform(X_vid_test_scaled)
        X_imu_test_pca = pca_imu.transform(X_imu_test_scaled)
        
        var_vid = np.sum(pca_vid.explained_variance_ratio_) * 100
        var_imu = np.sum(pca_imu.explained_variance_ratio_) * 100
        
        cca = CCA(n_components=1)
        cca.fit(X_vid_train_pca, X_imu_train_pca)
        
        X_c_test, Y_c_test = cca.transform(X_vid_test_pca, X_imu_test_pca)
        correlation_test = np.corrcoef(X_c_test[:, 0], Y_c_test[:, 0])[0, 1]
        
        results.append((k, var_vid, var_imu, correlation_test))
        table.add_row(str(k), f"{var_vid:.2f}%", f"{var_imu:.2f}%", f"{correlation_test:.4f}")
        
    console.print(table)
    
    report_path = os.path.join(RESULTS_DIR, "cca_sensitivity_report.txt")
    with open(report_path, "w") as f:
        f.write("========================================================================\n")
        f.write(" CROSS-SESSION PCA-CCA SENSITIVITY ANALYSIS: VIDEO VS IMU FEATURES \n")
        f.write("========================================================================\n\n")
        
        f.write("1. DATASET CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Strict Cross-Session Protocol (NO FLIPS)\n")
        f.write(f"Train Samples (FirstRun, SecondRun): {len(X_vid_train)}\n")
        f.write(f"Test Samples (ThirdRun): {len(X_vid_test)}\n")
        f.write(f"Original Video Dimensionality (Depth+RGB+IR): {X_vid_train.shape[1]} features\n")
        f.write(f"Original IMU Dimensionality: {X_imu_train.shape[1]} features\n\n")
        
        f.write("2. METHODOLOGY RATIONALE\n")
        f.write("-" * 40 + "\n")
        f.write("To prevent artificial 1.0 correlations due to overfitting and the P >> N\n")
        f.write("problem, the dataset was strictly split temporally (Cross-Session).\n")
        f.write("PCA and CCA models were fitted strictly on the Training subset (Run 1 & 2).\n")
        f.write("The reported Canonical Correlation reflects the out-of-sample projection\n")
        f.write("on the unseen Test subset (Run 3), empirically measuring the true semantic gap.\n\n")
        
        f.write("3. RESULTS (OUT-OF-SAMPLE TEST)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'PCA Components':<15} | {'Video Retained Var.':<20} | {'IMU Retained Var.':<20} | {'Test Correlation':<15}\n")
        f.write("-" * 78 + "\n")
        for r in results:
            f.write(f"{r[0]:<15} | {r[1]:>18.2f}% | {r[2]:>18.2f}% | {r[3]:>15.4f}\n")
        
    console.print(f"\n[bold green]Report successfully generated at: {report_path}[/bold green]")

if __name__ == "__main__":
    run_pca_cca_sensitivity()
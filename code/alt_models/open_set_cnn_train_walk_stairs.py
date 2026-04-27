import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
RESULTS_DIR = "results/alt_models/cnn_walk_stairs/open_set"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_ALL_VALID_AS_KNOWN = False
NUM_KNOWN_SUBJECTS = 15
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

class MultimodalGaitCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalGaitCNN, self).__init__()
        self.video_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=31, stride=4, padding=15), nn.ReLU(), nn.BatchNorm1d(16), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=15, stride=4, padding=7), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64)
        )
        self.imu_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7), nn.ReLU(), nn.BatchNorm1d(16), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        v = self.video_branch(x[:, :, :73728])
        if v.device.type == 'mps': v = nn.functional.adaptive_avg_pool1d(v.cpu(), 30).to('mps')
        else: v = nn.functional.adaptive_avg_pool1d(v, 30)
        
        i = self.imu_branch(x[:, :, 73728:])
        if i.device.type == 'mps': i = nn.functional.adaptive_avg_pool1d(i.cpu(), 20).to('mps')
        else: i = nn.functional.adaptive_avg_pool1d(i, 20)
        
        return self.classifier(torch.cat((v.flatten(1), i.flatten(1)), dim=1))

def get_subject_pools():
    if not os.path.exists(FEATURES_ROOT): return None, None
    all_subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    valid_subjects = []
    for subj in all_subjects:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        if not (os.path.exists(os.path.join(subj_path, "FirstRun")) and 
                os.path.exists(os.path.join(subj_path, "SecondRun")) and
                os.path.exists(os.path.join(subj_path, "ThirdRun"))): continue
            
        is_complete = True
        for run_name in ["FirstRun", "SecondRun", "ThirdRun"]:
            run_dir = os.path.join(subj_path, run_name)
            walk_count, stairs_count = 0, 0
            for root, _, files in os.walk(run_dir):
                root_lower = root.lower()
                if 'debug' in root_lower or "slope" in root_lower or "_backup" in root_lower: continue
                for f in files:
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        f_lower = f.lower()
                        if 'walk' in f_lower: walk_count += 1
                        elif 'stairs' in f_lower or 'up' in f_lower or 'down' in f_lower: stairs_count += 1
            if walk_count < 6 or stairs_count < 6: is_complete = False; break
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
    X_train, y_train_str, X_genuine, y_genuine_str, X_impostor = [], [], [], [], [] 
    
    for subj in known_subs:
        for root, _, files in os.walk(os.path.join(FEATURES_ROOT, subj)):
            if 'debug' in root.lower() or "slope" in root.lower() or "_backup" in root.lower(): continue
            for f in files:
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: vec = np.load(os.path.join(root, f))
                except: continue
                if 'FirstRun' in root or 'SecondRun' in root: X_train.append(vec); y_train_str.append(subj)
                elif 'ThirdRun' in root: X_genuine.append(vec); y_genuine_str.append(subj)

    for subj in unknown_subs:
        for root, _, files in os.walk(os.path.join(FEATURES_ROOT, subj)):
            if 'debug' in root.lower() or "slope" in root.lower() or "_backup" in root.lower(): continue
            for f in files:
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: vec = np.load(os.path.join(root, f))
                except: continue
                X_impostor.append(vec)

    return np.nan_to_num(X_train), np.array(y_train_str), np.nan_to_num(X_genuine), np.array(y_genuine_str), np.nan_to_num(X_impostor)

def calculate_metrics(probs_genuine, y_genuine, probs_impostor, thresholds):
    TG, TI = len(probs_genuine), len(probs_impostor)
    DIR_list, FAR_list, FRR_list = [], [], []
    for t in thresholds:
        DI = sum(1 for i in range(TG) if np.max(probs_genuine[i]) >= t and np.argmax(probs_genuine[i]) == y_genuine[i])
        FA = sum(1 for i in range(TI) if np.max(probs_impostor[i]) >= t)
        DIR_list.append(DI / TG if TG > 0 else 0)
        FAR_list.append(FA / TI if TI > 0 else 0)
        FRR_list.append(1 - (DI / TG) if TG > 0 else 1) 
    return np.array(DIR_list), np.array(FAR_list), np.array(FRR_list)

def main():
    console.print(Panel.fit("[bold cyan]OPEN SET PROTOCOL: CNN 1D (Walk/Stairs)[/bold cyan]"))
    known, unknown = get_subject_pools()
    if not known: return

    console.print(f"Gallery (Known): {len(known)} subjects")
    console.print(f"Impostors (Unknown): {len(unknown)} subjects")
    
    X_train, y_train_str, X_gen, y_gen_str, X_imp = load_data(known, unknown)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_gen = le.transform(y_gen_str)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_gen = scaler.transform(X_gen)
    if len(X_imp) > 0: X_imp = scaler.transform(X_imp)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalGaitCNN(num_classes=len(known)).to(device)

    model_name = "cnn_walk_stairs_closed_set.pth" if USE_ALL_VALID_AS_KNOWN else "cnn_walk_stairs_open_set.pth"
    model_path = os.path.join(MODEL_DIR, model_name)

    if os.path.exists(model_path):
        console.print(f"[bold green]Loading existing model weights: {model_name}...[/bold green]")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        console.print("[bold yellow]Training CNN from scratch...[/bold yellow]")
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), model_path)
        console.print(f"[bold green]Model saved to {model_name}![/bold green]")

    console.print(f"Testing on {len(X_gen)} Genuine Probes and {len(X_imp)} Impostor Probes...")
    model.eval()
    with torch.no_grad():
        outputs_gen = model(torch.tensor(X_gen, dtype=torch.float32).unsqueeze(1).to(device))
        probs_gen = torch.softmax(outputs_gen, dim=1).cpu().numpy()
        
        if len(X_imp) > 0:
            outputs_imp = model(torch.tensor(X_imp, dtype=torch.float32).unsqueeze(1).to(device))
            probs_imp = torch.softmax(outputs_imp, dim=1).cpu().numpy()
        else:
            probs_imp = np.array([])

    thresholds = np.linspace(0.0, 1.0, 200)
    DIR, FAR, FRR = calculate_metrics(probs_gen, y_gen, probs_imp, thresholds)

    eer_threshold = brentq(lambda t: interp1d(thresholds, FAR)(t) - interp1d(thresholds, FRR)(t), 0.0, 1.0)
    eer_value = float(interp1d(thresholds, FAR)(eer_threshold))
    dir_at_eer = float(interp1d(thresholds, DIR)(eer_threshold))

    sort_idx = np.argsort(FAR)
    roc_auc = auc(FAR[sort_idx], DIR[sort_idx])

    table = Table(title="Open Set Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan"); table.add_column("Value", justify="right", style="green")
    table.add_row("Equal Error Rate (EER)", f"{eer_value*100:.2f}%"); table.add_row("Balance Threshold", f"{eer_threshold:.2f}")
    console.print(table)

    suffix = "_19_known" if USE_ALL_VALID_AS_KNOWN else "_15_known"
    
    plt.figure(figsize=(10, 6))
    plt.plot(FAR, DIR, label='Watchlist ROC', color='teal', linewidth=2)
    plt.plot(eer_value, dir_at_eer, 'ro', markersize=8, label=f'EER Point ({eer_value*100:.1f}%)')
    plt.title('Watchlist ROC (Open Set) - CNN Walk/Stairs')
    plt.xlabel('False Alarm Rate (FAR)'); plt.ylabel('Detect & Identify Rate (DIR)')
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05]); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f'roc_watchlist_cnn{suffix}.png'), dpi=300); plt.close()

    with open(os.path.join(RESULTS_DIR, f"open_set_report{suffix}.txt"), "w") as f:
        f.write("========================================================\n")
        f.write(" OPEN SET WATCHLIST REPORT: CNN Walk/Stairs\n")
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

if __name__ == "__main__":
    main()
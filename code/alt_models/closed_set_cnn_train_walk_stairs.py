import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
RESULTS_DIR = "results/cnn_walk_stairs/closed_set"

SAVE_RESULTS = False
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
SEED = 42

if SAVE_RESULTS:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data():
    console.print(Panel.fit(f"[bold yellow]Loading Walk/Stairs Dataset for PyTorch CNN: {FEATURES_ROOT}[/bold yellow]"))
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
                if not f.endswith('.npy') or f.startswith('._'): continue
                file_path = os.path.join(root, f)
                try: vector = np.load(file_path)
                except: continue
                
                if 'FirstRun' in file_path or 'SecondRun' in file_path:
                    X_train.append(vector); y_train_str.append(subj)
                elif 'ThirdRun' in file_path:
                    X_test.append(vector); y_test_str.append(subj); meta_test.append(f)
                    
    return np.array(X_train), np.array(X_test), np.array(y_train_str), np.array(y_test_str), meta_test

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
        if v.device.type == 'mps':
            v = nn.functional.adaptive_avg_pool1d(v.cpu(), 30).to('mps')
        else: v = nn.functional.adaptive_avg_pool1d(v, 30)
        
        i = self.imu_branch(x[:, :, 73728:])
        if i.device.type == 'mps':
            i = nn.functional.adaptive_avg_pool1d(i.cpu(), 20).to('mps')
        else: i = nn.functional.adaptive_avg_pool1d(i, 20)
        
        return self.classifier(torch.cat((v.flatten(1), i.flatten(1)), dim=1))

def train_and_evaluate():
    data = load_data()
    if data is None: return
    X_train, X_test, y_train_str, y_test_str, meta_test = data
    
    if np.isnan(X_train).any(): X_train, X_test = np.nan_to_num(X_train), np.nan_to_num(X_test)
        
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_test = label_encoder.transform(y_test_str)
    classes = label_encoder.classes_
    
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalGaitCNN(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    console.print(Panel("[bold green]Starting Training Loop...[/bold green]"))
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total += labels.size(0)
        if (epoch+1) % 5 == 0 or epoch == 0:
            console.print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {100*correct/total:.2f}%")

    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device))
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_preds = torch.max(outputs, 1)[1].cpu().numpy()

    y_pred_str_out = label_encoder.inverse_transform(all_preds)
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
            f.write("=== FINAL REPORT: CNN (WALK/STAIRS) ===\n\n--- CLASSIFICATION REPORT ---\n")
            f.write(report_text + "\n\n--- CUMULATIVE MATCH SCORE (CMS) ---\n")
            for k in [1, 2, 3, 4, 5, 10]:
                if k <= G: f.write(f"Rank-{k}: {cms[k-1]:.2f}%\n")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, G + 1), cms, marker='o', linestyle='-', color='teal', linewidth=2)
        plt.title('Cumulative Match Characteristic (CMC) - CNN Walk/Stairs')
        plt.xlabel('Rank'); plt.ylabel('Recognition Rate (%)'); plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(np.arange(1, min(G + 1, 21), 1)); plt.ylim(0, 105); plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'cmc_curve.png'), dpi=300); plt.close()

        cm = confusion_matrix(y_test_str, y_pred_str_out, labels=classes)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - CNN Walk/Stairs')
        plt.ylabel('True Identity (Probe)'); plt.xlabel('Predicted Identity (Gallery Match)')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300); plt.close()

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "cnn_walk_stairs.pth"))

if __name__ == "__main__":
    train_and_evaluate()
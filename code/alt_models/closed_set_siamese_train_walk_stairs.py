import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models/alt_models"
RESULTS_DIR = "results/alt_models/siamese_walk_stairs/no_flip/closed_set"

SAVE_RESULTS = False
BATCH_SIZE = 16 
EPOCHS = 40
LEARNING_RATE = 0.0005
MARGIN = 2.0 
SEED = 42

if SAVE_RESULTS:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def load_data():
    console.print(Panel.fit(f"[bold yellow]Loading Walk/Stairs for SIAMESE NETWORK: {FEATURES_ROOT}[/bold yellow]"))
    X_train, y_train_str, X_test, y_test_str = [], [], [], []
    
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
            if walk_count < 6 or stairs_count < 6:
                is_complete = False; break
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
                    X_train.append(vector)
                    y_train_str.append(subj)
                elif 'ThirdRun' in file_path:
                    X_test.append(vector)
                    y_test_str.append(subj)
                    
    return np.array(X_train), np.array(X_test), np.array(y_train_str), np.array(y_test_str)

class SiameseTrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y
        self.classes = np.unique(y)
        self.class_indices = {cls: np.where(self.y == cls)[0] for cls in self.classes}
        
    def __len__(self):
        return len(self.X) * 3 

    def __getitem__(self, index):
        same_person = random.random() > 0.5
        
        if same_person:
            cls = random.choice(self.classes)
            idx1, idx2 = random.sample(list(self.class_indices[cls]), 2)
            img1, img2 = self.X[idx1], self.X[idx2]
            label = torch.tensor(1.0, dtype=torch.float32) 
        else:
            cls1, cls2 = random.sample(list(self.classes), 2)
            idx1 = random.choice(self.class_indices[cls1])
            idx2 = random.choice(self.class_indices[cls2])
            img1, img2 = self.X[idx1], self.X[idx2]
            label = torch.tensor(0.0, dtype=torch.float32) 
            
        return img1, img2, label

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(78658, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.6), 
            nn.Linear(4096, 1024) 
        )

    def forward(self, x):
        return self.model(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

def train_and_evaluate():
    data = load_data()
    if data is None: return
    X_train, X_test, y_train_str, y_test_str = data
    
    if np.isnan(X_train).any():
        X_train, X_test = np.nan_to_num(X_train), np.nan_to_num(X_test)
        
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_test = label_encoder.transform(y_test_str)
    classes = label_encoder.classes_
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = SiameseTrainDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using device: {device}[/bold cyan]")
    
    model = EmbeddingNet().to(device)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    console.print(Panel("[bold green]Starting SIAMESE Training Loop...[/bold green]"))
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            out1 = model(img1)
            out2 = model(img2)
            
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if (epoch+1) % 5 == 0 or epoch == 0:
            console.print(f"Epoch [{epoch+1}/{EPOCHS}] - Contrastive Loss: {running_loss/len(train_loader):.4f}")

    console.print(Panel("[bold blue]Extracting Embeddings for Gallery & Probe...[/bold blue]"))
    model.eval()
    
    X_gallery_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        gallery_embeddings = model(X_gallery_tensor)
        
    X_probe_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        probe_embeddings = model(X_probe_tensor)

    all_preds, all_probs = [], []
    
    for i in range(len(probe_embeddings)):
        probe_emb = probe_embeddings[i].unsqueeze(0) 
        distances = nn.functional.pairwise_distance(probe_emb, gallery_embeddings).cpu().numpy()
        
        sorted_indices = np.argsort(distances)
        predicted_identity = y_train[sorted_indices[0]]
        all_preds.append(predicted_identity)
        
        class_scores = np.zeros(len(classes))
        for gal_idx, dist in enumerate(distances):
            gal_class = y_train[gal_idx]
            class_scores[gal_class] = max(class_scores[gal_class], 1.0 / (dist + 1e-6))
            
        all_probs.append(class_scores)

    y_pred_str_out = label_encoder.inverse_transform(all_preds)
    
    report_text = classification_report(y_test_str, y_pred_str_out)
    console.print(report_text)

    TA, G = len(y_test), len(classes)
    ranks_counts = np.zeros(G)
    for i in range(TA):
        sorted_indices = np.argsort(all_probs[i])[::-1]
        ranks_counts[np.where(sorted_indices == y_test[i])[0][0]] += 1
    cms = np.cumsum(ranks_counts) / TA * 100

    cmc_table = Table(title="Closed Set Metrics (CMC)", box=box.ROUNDED)
    cmc_table.add_column("Rank", justify="center", style="magenta")
    cmc_table.add_column("Cumulative Match Score", justify="center", style="green")
    for k in [1, 2, 3, 4, 5, 10]:
        if k <= G: cmc_table.add_row(f"Rank-{k}", f"{cms[k-1]:.2f}%")
    console.print(cmc_table)

    if SAVE_RESULTS:
        with open(os.path.join(RESULTS_DIR, "metrics_report.txt"), "w") as f:
            f.write("=== FINAL REPORT: SIAMESE NETWORK (WALK/STAIRS) ===\n\n--- CLASSIFICATION REPORT ---\n")
            f.write(report_text + "\n\n--- CUMULATIVE MATCH SCORE (CMS) ---\n")
            for k in [1, 2, 3, 4, 5, 10]:
                if k <= G: f.write(f"Rank-{k}: {cms[k-1]:.2f}%\n")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, G + 1), cms, marker='o', linestyle='-', color='teal', linewidth=2)
        plt.title('Cumulative Match Characteristic (CMC) - Siamese Walk/Stairs')
        plt.xlabel('Rank'); plt.ylabel('Recognition Rate (%)'); plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(np.arange(1, min(G + 1, 21), 1)); plt.ylim(0, 105); plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'cmc_curve.png'), dpi=300); plt.close()

        cm = confusion_matrix(y_test_str, y_pred_str_out, labels=classes)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix - Siamese Walk/Stairs')
        plt.ylabel('True Identity (Probe)'); plt.xlabel('Predicted Identity (Gallery Match)')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300); plt.close()

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "siamese_walk_stairs_closed_set.pth"))

if __name__ == "__main__":
    train_and_evaluate()
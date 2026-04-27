import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import auc
import warnings
import random
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models/alt_models"
RESULTS_DIR = "results/alt_models/siamese_slope/open_set"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_ALL_VALID_AS_KNOWN = False

NUM_KNOWN_SUBJECTS = 15
VIDEO_CUT_INDEX = 73728
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.0005
MARGIN = 2.0
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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
            slope_count = 0
            for root, _, files in os.walk(run_dir):
                root_lower = root.lower()
                if 'debug' in root_lower or "slope" not in root_lower: continue
                for f in files:
                    if f.endswith('.npy') and 'flip' not in f and not f.startswith('._'):
                        slope_count += 1
            if slope_count < 6: is_complete = False; break
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
            if 'debug' in root.lower() or "slope" not in root.lower(): continue
            for f in files:
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                    if len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
                    else: continue
                except: continue
                if 'FirstRun' in root or 'SecondRun' in root: X_train.append(vec); y_train_str.append(subj)
                elif 'ThirdRun' in root: X_genuine.append(vec); y_genuine_str.append(subj)

    for subj in unknown_subs:
        for root, _, files in os.walk(os.path.join(FEATURES_ROOT, subj)):
            if 'debug' in root.lower() or "slope" not in root.lower(): continue
            for f in files:
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                    if len(vec) > VIDEO_CUT_INDEX: vec = vec[VIDEO_CUT_INDEX:]
                    else: continue
                except: continue
                X_impostor.append(vec)

    return np.nan_to_num(X_train), np.array(y_train_str), np.nan_to_num(X_genuine), np.array(y_genuine_str), np.nan_to_num(X_impostor)

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
    def __init__(self, input_dim):
        super(EmbeddingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256)
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
    console.print(Panel.fit("[bold magenta]OPEN SET PROTOCOL: Siamese (Slope)[/bold magenta]"))
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
    input_dim = X_train.shape[1]
    model = EmbeddingNet(input_dim).to(device)

    model_name = "siamese_slope_closed_set.pth" if USE_ALL_VALID_AS_KNOWN else "siamese_slope_open_set.pth"
    model_path = os.path.join(MODEL_DIR, model_name)

    if os.path.exists(model_path):
        console.print(f"[bold green]Loading existing model weights: {model_name}...[/bold green]")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        console.print("[bold yellow]Training Siamese Network from scratch...[/bold yellow]")
        train_dataset = SiameseTrainDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        criterion = ContrastiveLoss(margin=MARGIN)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
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
                
        torch.save(model.state_dict(), model_path)
        console.print(f"[bold green]Model saved to {model_name}![/bold green]")

    console.print(f"Testing on {len(X_gen)} Genuine Probes and {len(X_imp)} Impostor Probes...")
    model.eval()
    
    X_gallery_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        gallery_embeddings = model(X_gallery_tensor)
        
    X_gen_tensor = torch.tensor(X_gen, dtype=torch.float32).to(device)
    with torch.no_grad():
        gen_embeddings = model(X_gen_tensor)

    probs_gen = []
    for i in range(len(gen_embeddings)):
        probe_emb = gen_embeddings[i].unsqueeze(0) 
        distances = nn.functional.pairwise_distance(probe_emb, gallery_embeddings).cpu().numpy()
        class_scores = np.zeros(len(known))
        for gal_idx, dist in enumerate(distances):
            gal_class = y_train[gal_idx]
            sim = np.exp(-dist) # Converte distanza in probabilità [0,1]
            class_scores[gal_class] = max(class_scores[gal_class], sim)
        probs_gen.append(class_scores)
    probs_gen = np.array(probs_gen)

    probs_imp = []
    if len(X_imp) > 0:
        X_imp_tensor = torch.tensor(X_imp, dtype=torch.float32).to(device)
        with torch.no_grad():
            imp_embeddings = model(X_imp_tensor)
        for i in range(len(imp_embeddings)):
            probe_emb = imp_embeddings[i].unsqueeze(0) 
            distances = nn.functional.pairwise_distance(probe_emb, gallery_embeddings).cpu().numpy()
            class_scores = np.zeros(len(known))
            for gal_idx, dist in enumerate(distances):
                gal_class = y_train[gal_idx]
                sim = np.exp(-dist)
                class_scores[gal_class] = max(class_scores[gal_class], sim)
            probs_imp.append(class_scores)
    probs_imp = np.array(probs_imp)

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
    plt.title('Watchlist ROC (Open Set) - Siamese Slope')
    plt.xlabel('False Alarm Rate (FAR)'); plt.ylabel('Detect & Identify Rate (DIR)')
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05]); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f'roc_watchlist_siamese{suffix}.png'), dpi=300); plt.close()

    with open(os.path.join(RESULTS_DIR, f"open_set_report{suffix}.txt"), "w") as f:
        f.write("========================================================\n")
        f.write(" OPEN SET WATCHLIST REPORT: Siamese Slope\n")
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
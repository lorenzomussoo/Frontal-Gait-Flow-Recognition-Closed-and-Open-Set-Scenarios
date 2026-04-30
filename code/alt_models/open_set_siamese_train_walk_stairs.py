import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import warnings
import random
from rich.console import Console
from rich.panel import Panel
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
MODEL_DIR = "models/alt_models"
RESULTS_DIR = "results/alt_models/siamese_walk_stairs/masked_open_set"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 16 
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
            walk_count, stairs_count = 0, 0
            for root, dirs, files in os.walk(run_dir):
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
    return sorted(valid_subjects), sorted(incomplete_subjects)

def load_data(valid_subs, incomplete_subs):
    X_train, y_train_str = [], []
    X_gen, y_gen_str = [], []
    X_imp = []
    
    for subj in valid_subs:
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            root_lower = root.lower()
            if 'debug' in root_lower or "_backup" in root_lower or "slope" in root_lower: continue
                
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
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
            if 'debug' in root_lower or "_backup" in root_lower or "slope" in root_lower: continue
                
            for f in sorted(files):
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                try: 
                    vec = np.load(os.path.join(root, f))
                except: continue
                X_imp.append(vec)
                
    return (np.nan_to_num(X_train), np.array(y_train_str), 
            np.nan_to_num(X_gen), np.array(y_gen_str), 
            np.nan_to_num(X_imp))

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
            nn.Linear(input_dim, 4096),
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

def calculate_operational_point(FAR_array, DIR_array, thresholds, target_far):
    idx = np.where(FAR_array <= target_far)[0]
    if len(idx) > 0:
        best_idx = idx[0]
        return thresholds[best_idx], DIR_array[best_idx], FAR_array[best_idx]
    return None, None, None

def run_analysis(valid_subs, incomplete_subs):
    X_train, y_train_str, X_gen, y_gen_str, X_imp = load_data(valid_subs, incomplete_subs)
    
    unique_known = sorted(list(set(y_train_str)))
    label_map = {name: idx for idx, name in enumerate(unique_known)}
    rev_label_map = {idx: name for name, idx in label_map.items()}
    y_train = np.array([label_map[s] for s in y_train_str])
    y_gen = np.array([label_map[s] for s in y_gen_str])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_gen_scaled = scaler.transform(X_gen)
    X_imp_scaled = scaler.transform(X_imp) if len(X_imp) > 0 else []

    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    model = EmbeddingNet(input_dim).to(device)

    model_path = os.path.join(MODEL_DIR, "siamese_walk_stairs_masked.pth")

    if os.path.exists(model_path):
        console.print(f"[bold green]Loading existing model weights: siamese_walk_stairs_masked.pth...[/bold green]")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        console.print("[bold yellow]Training Siamese Network from scratch...[/bold yellow]")
        train_dataset = SiameseTrainDataset(X_train_scaled, y_train)
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
        console.print(f"[bold green]Model saved to siamese_walk_stairs_masked.pth![/bold green]")

        console.print("Extracting embeddings and probabilities...")
    model.eval()
    
    def get_embeddings_in_batches(data_array, batch_size=16):
        embeds = []
        with torch.no_grad():
            for i in range(0, len(data_array), batch_size):
                batch_tensor = torch.tensor(data_array[i:i+batch_size], dtype=torch.float32).to(device)
                batch_emb = model(batch_tensor)
                embeds.append(batch_emb)
        return torch.cat(embeds, dim=0)

    gallery_embeddings = get_embeddings_in_batches(X_train_scaled)
    gen_embeddings = get_embeddings_in_batches(X_gen_scaled)

    probs_gen = []
    for i in range(len(gen_embeddings)):
        probe_emb = gen_embeddings[i].unsqueeze(0) 
        distances = nn.functional.pairwise_distance(probe_emb, gallery_embeddings).cpu().numpy()
        class_scores = np.zeros(len(unique_known))
        for gal_idx, dist in enumerate(distances):
            gal_class = y_train[gal_idx]
            sim = np.exp(-dist) 
            class_scores[gal_class] = max(class_scores[gal_class], sim)
        probs_gen.append(class_scores)
    probs_gen = np.array(probs_gen)

    probs_imp = []
    if len(X_imp_scaled) > 0:
        imp_embeddings = get_embeddings_in_batches(X_imp_scaled)
        for i in range(len(imp_embeddings)):
            probe_emb = imp_embeddings[i].unsqueeze(0) 
            distances = nn.functional.pairwise_distance(probe_emb, gallery_embeddings).cpu().numpy()
            class_scores = np.zeros(len(unique_known))
            for gal_idx, dist in enumerate(distances):
                gal_class = y_train[gal_idx]
                sim = np.exp(-dist)
                class_scores[gal_class] = max(class_scores[gal_class], sim)
            probs_imp.append(class_scores)
    probs_imp = np.array(probs_imp)
    
    TG = len(probs_gen)
    TI_virtual = len(probs_gen) 
    TI_true = len(probs_imp)
    TI_total = TI_virtual + TI_true 
    
    thresholds = np.linspace(0.0, 1.0, 500)
    DIR_list, FAR_list, FRR_list = [], [], []
    
    genuine_confidences = []
    impostor_confidences = []
    subject_confidences = {subj: [] for subj in unique_known}
    
    for i in range(TG):
        if np.argmax(probs_gen[i]) == y_gen[i]:
            conf = np.max(probs_gen[i])
            genuine_confidences.append(conf)
            subject_confidences[rev_label_map[y_gen[i]]].append(conf)
            
        masked_probs = probs_gen[i].copy()
        masked_probs[y_gen[i]] = -1.0 
        impostor_confidences.append(np.max(masked_probs))
        
    for i in range(TI_true):
        impostor_confidences.append(np.max(probs_imp[i]))
        
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
        
    DIR_array, FAR_array, FRR_array = np.array(DIR_list), np.array(FAR_list), np.array(FRR_list)
    
    def eer_func(t):
        return interp1d(thresholds, FAR_array)(t) - interp1d(thresholds, FRR_array)(t)

    try:
        eer_threshold = brentq(eer_func, 0.0, 1.0)
        eer_value = float(interp1d(thresholds, FAR_array)(eer_threshold))
        dir_at_eer = float(interp1d(thresholds, DIR_array)(eer_threshold))
    except ValueError:
        idx_eer = np.argmin(np.abs(FAR_array - FRR_array))
        eer_threshold = thresholds[idx_eer]
        eer_value = FAR_array[idx_eer]
        dir_at_eer = DIR_array[idx_eer]
    
    sort_idx = np.argsort(FAR_array)
    roc_auc = auc(FAR_array[sort_idx], DIR_array[sort_idx])
    
    t_1, dir_1, far_1 = calculate_operational_point(FAR_array, DIR_array, thresholds, 0.01)
    t_5, dir_5, far_5 = calculate_operational_point(FAR_array, DIR_array, thresholds, 0.05)
    t_10, dir_10, far_10 = calculate_operational_point(FAR_array, DIR_array, thresholds, 0.10)

    plt.figure(figsize=(9, 6))
    plt.plot(FAR_array, DIR_array, label=f'Watchlist ROC (AUC = {roc_auc:.4f})', color='magenta', linewidth=2.5)
    plt.plot(eer_value, dir_at_eer, 'ro', markersize=8, label=f'EER Point ({eer_value*100:.2f}%)')
    plt.title('Masked Open Set ROC - Siamese (Walk/Stairs)', fontsize=14, pad=15)
    plt.xlabel('False Alarm Rate (FAR)', fontsize=12)
    plt.ylabel('Detect & Identify Rate (DIR)', fontsize=12)
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_siamese_walk_stairs.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(genuine_confidences, fill=True, color="green", label="Genuine Attempts (Correct ID)", alpha=0.5)
    sns.kdeplot(impostor_confidences, fill=True, color="red", label="Impostor Attempts (Masked + Strangers)", alpha=0.5)
    plt.title('Score Distribution - Siamese (Walk/Stairs)', fontsize=14, pad=15)
    plt.xlabel('Confidence Score (Probability)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'dist_siamese_walk_stairs.png'), dpi=300)
    plt.close()
    
    report_path = os.path.join(RESULTS_DIR, "siamese_walk_stairs_comprehensive_report.txt")
    with open(report_path, "w") as f:
        f.write("==================================================================\n")
        f.write(" COMPREHENSIVE OPEN SET REPORT - SIAMESE (WALK/STAIRS)\n")
        f.write("==================================================================\n\n")
        f.write("1. DATASET COMPOSITION:\n")
        f.write(f"   - Registered Subjects (Gallery): {len(valid_subs)}\n")
        f.write(f"   - Unknown Subjects (True Impostors): {len(incomplete_subs)}\n")
        f.write(f"   - Total Genuine Attempts (TG): {TG}\n")
        f.write(f"   - Total Impostor Attempts (TI): {TI_total} ({TI_virtual} Virtual + {TI_true} Strangers)\n\n")
        
        f.write("2. GLOBAL METRICS:\n")
        f.write(f"   - Equal Error Rate (EER): {eer_value*100:.2f}%\n")
        f.write(f"   - EER Balance Threshold: {eer_threshold:.4f}\n")
        f.write(f"   - FAR at EER Point: {eer_value*100:.2f}%\n")
        f.write(f"   - FRR at EER Point: {eer_value*100:.2f}%\n")
        f.write(f"   - Genuine Reject Rate (GRR) at EER: {100 - (eer_value*100):.2f}%\n")
        f.write(f"   - ROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("3. OPERATIONAL POINTS (HIGH SECURITY):\n")
        if t_1 is not None:
            f.write("   Maximum Allowable FAR: 1.0%\n")
            f.write(f"     -> Threshold Required: >= {t_1:.4f}\n")
            f.write(f"     -> Resulting DIR: {dir_1*100:.2f}%\n")
            f.write(f"     -> Actual FAR: {far_1*100:.2f}%\n\n")
        
        if t_5 is not None:
            f.write("   Maximum Allowable FAR: 5.0%\n")
            f.write(f"     -> Threshold Required: >= {t_5:.4f}\n")
            f.write(f"     -> Resulting DIR: {dir_5*100:.2f}%\n")
            f.write(f"     -> Actual FAR: {far_5*100:.2f}%\n\n")
        
        if t_10 is not None:
            f.write("   Maximum Allowable FAR: 10.0%\n")
            f.write(f"     -> Threshold Required: >= {t_10:.4f}\n")
            f.write(f"     -> Resulting DIR: {dir_10*100:.2f}%\n")
            f.write(f"     -> Actual FAR: {far_10*100:.2f}%\n\n")
        
        f.write("4. CONFIDENCE ANALYSIS:\n")
        f.write(f"   - Overall Average Genuine Confidence: {np.mean(genuine_confidences):.4f} (std: {np.std(genuine_confidences):.4f})\n")
        f.write(f"   - Overall Average Impostor Confidence: {np.mean(impostor_confidences):.4f} (std: {np.std(impostor_confidences):.4f})\n\n")
        f.write("   Average Genuine Confidence per Subject (Gallery 1-19):\n")
        for subj, confs in subject_confidences.items():
            if len(confs) > 0:
                f.write(f"     * {subj}: {np.mean(confs):.4f} (based on {len(confs)} correct predictions)\n")
            else:
                f.write(f"     * {subj}: 0.0000 (0 correct predictions)\n")
                
    console.print(f"\n[bold green]Success! Analysis finished.[/bold green]")
    console.print(f"Metrics: EER = {eer_value*100:.2f}%, AUC = {roc_auc:.4f}")
    console.print(f"Results saved in: [cyan]{RESULTS_DIR}[/cyan]")

def main():
    console.print(Panel.fit("[bold magenta]COMPREHENSIVE OPEN SET ANALYSIS - SIAMESE WALK/STAIRS[/bold magenta]"))
    valid_subs, incomplete_subs = get_subject_pools()
    if not valid_subs: 
        console.print("[red]No valid subjects found. Check paths.[/red]")
        return
    run_analysis(valid_subs, incomplete_subs)

if __name__ == "__main__":
    main()
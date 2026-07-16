import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
RESULTS_DIR = "results/CCA"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

L = 24576
IDX_VIDEO_END = 3 * L 
IDX_IMU_START = 3 * L 

def load_labeled_data_no_flips():
    console.print(Panel.fit("[bold cyan]Loading Dataset for t-SNE Visualization (NO FLIPS)[/bold cyan]"))
    X_video, X_imu, y_subj, y_task = [], [], [], []
    
    if not os.path.exists(FEATURES_ROOT):
        console.print("[red]Error: Features folder not found![/red]")
        return None, None, None, None

    all_subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    for subj in track(all_subjects, description="Loading pristine subjects..."): 
        subj_path = os.path.join(FEATURES_ROOT, subj)
        for root, dirs, files in os.walk(subj_path):
            if 'debug' in root or "_backup" in root.lower() or "slope" in root.lower(): continue
            for f in files:
                if not f.endswith('.npy') or f.startswith('._') or 'flip' in f: continue
                
                file_path = os.path.join(root, f)
                task = 'Walk' if 'walk' in f.lower() else 'Stairs'
                
                try: 
                    vector = np.load(file_path)
                    X_video.append(vector[:IDX_VIDEO_END])
                    X_imu.append(vector[IDX_IMU_START:])
                    y_subj.append(subj)
                    y_task.append(task)
                except: continue
                    
    X_video = np.nan_to_num(np.array(X_video))
    X_imu = np.nan_to_num(np.array(X_imu))
    y_subj = np.array(y_subj)
    y_task = np.array(y_task)
    
    console.print(f"\n[bold green]Loaded {len(X_video)} pristine samples.[/bold green]")
    return X_video, X_imu, y_subj, y_task

def run_tsne_visualization():
    X_video, X_imu, y_subj, y_task = load_labeled_data_no_flips()
    if X_video is None: return

    console.print("[yellow]1. Standardizing features...[/yellow]")
    X_vid_scaled = StandardScaler().fit_transform(X_video)
    X_imu_scaled = StandardScaler().fit_transform(X_imu)

    console.print("[yellow]2. Applying PCA (50 components) for noise reduction prior to t-SNE...[/yellow]")
    pca_vid = PCA(n_components=50, random_state=SEED).fit_transform(X_vid_scaled)
    pca_imu = PCA(n_components=50, random_state=SEED).fit_transform(X_imu_scaled)

    console.print("[yellow]3. Computing t-SNE projections (this might take a minute)...[/yellow]")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init='pca', learning_rate='auto')
    
    vid_embedded = tsne.fit_transform(pca_vid)
    imu_embedded = tsne.fit_transform(pca_imu)

    console.print("[yellow]4. Generating Publication-Ready Plot...[/yellow]")
    
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    palette = sns.color_palette("tab20", n_colors=len(np.unique(y_subj)))

    sns.scatterplot(
        x=vid_embedded[:, 0], y=vid_embedded[:, 1],
        hue=y_subj, style=y_task, palette=palette, 
        s=60, alpha=0.8, ax=axes[0], legend=False
    )
    axes[0].set_title("A) Video Features (Depth+RGB+IR)\nLoss of Identity Signatures", fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)

    sns.scatterplot(
        x=imu_embedded[:, 0], y=imu_embedded[:, 1],
        hue=y_subj, style=y_task, palette=palette, 
        s=60, alpha=0.8, ax=axes[1]
    )
    axes[1].set_title("B) IMU Features\nPreserved Identity Manifolds", fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=12)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=12)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title="Subject ID & Task", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize=11)

    plt.suptitle("t-SNE Manifold Projection: The Semantic Gap in Frontal Gait", fontsize=18, fontweight='black', y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, 'tsne_manifold_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"[bold green]Done! High-res plot saved to: {plot_path}[/bold green]")

if __name__ == "__main__":
    run_tsne_visualization()
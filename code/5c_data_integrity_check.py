import os
import numpy as np
import hashlib
from collections import defaultdict
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"

def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        vector = np.load(filepath)
        hasher.update(vector.tobytes())
        return hasher.hexdigest()
    except Exception:
        return None

def is_expected_flip_pair(paths):
    if len(paths) != 2:
        return False
    p1, p2 = paths[0], paths[1]
    p1_flipped = p1.replace('.npy', '_flip.npy')
    p2_flipped = p2.replace('.npy', '_flip.npy')
    return p1_flipped == p2 or p2_flipped == p1

def main():
    console.print(Panel.fit(f"[bold cyan]Starting 3-Way Data Integrity & Leakage Check: {FEATURES_ROOT}[/bold cyan]"))
    
    if not os.path.exists(FEATURES_ROOT):
        console.print("[red]Error: Features folder not found![/red]")
        return

    all_subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    valid_subjects = [s for s in all_subjects if os.path.exists(os.path.join(FEATURES_ROOT, s, "FirstRun")) and 
                                                 os.path.exists(os.path.join(FEATURES_ROOT, s, "SecondRun")) and
                                                 os.path.exists(os.path.join(FEATURES_ROOT, s, "ThirdRun"))]

    total_data_leaks = 0
    global_hashes = defaultdict(list)
    
    table = Table(title="Cross-Session Leakage Analysis per Subject")
    table.add_column("Subject", style="cyan")
    table.add_column("FirstRun Files", justify="right")
    table.add_column("SecondRun Files", justify="right")
    table.add_column("ThirdRun Files", justify="right")
    table.add_column("Leakage (Cross-Session)", justify="right", style="red")

    for subj in track(valid_subjects, description="Scanning files and computing hashes..."):
        runs = {"FirstRun": set(), "SecondRun": set(), "ThirdRun": set()}
        
        for run_name in runs.keys():
            run_dir = os.path.join(FEATURES_ROOT, subj, run_name)
            for root, _, files in os.walk(run_dir):
                if 'debug' in root: continue
                for f in files:
                    if f.endswith('.npy') and not f.startswith('._'):
                        file_path = os.path.join(root, f)
                        f_hash = get_file_hash(file_path)
                        if f_hash:
                            runs[run_name].add(f_hash)
                            global_hashes[f_hash].append(file_path)

        leak_1_2 = runs["FirstRun"].intersection(runs["SecondRun"])
        leak_1_3 = runs["FirstRun"].intersection(runs["ThirdRun"])
        leak_2_3 = runs["SecondRun"].intersection(runs["ThirdRun"])
        
        subj_leaks = len(leak_1_2) + len(leak_1_3) + len(leak_2_3)
        total_data_leaks += subj_leaks
        
        leak_str = f"[bold red]{subj_leaks}[/bold red]" if subj_leaks > 0 else "[green]0[/green]"
        
        table.add_row(subj, str(len(runs["FirstRun"])), str(len(runs["SecondRun"])), str(len(runs["ThirdRun"])), leak_str)

    console.print(table)

    console.print("\n[bold yellow]Performing Global Exact Duplicate Analysis...[/bold yellow]")
    
    exact_duplicates = {}
    expected_flip_pairs = 0
    
    for h, paths in global_hashes.items():
        if len(paths) > 1:
            if is_expected_flip_pair(paths):
                expected_flip_pairs += 1
            else:
                exact_duplicates[h] = paths
    
    if expected_flip_pairs > 0:
        console.print(f"[dim]Ignored {expected_flip_pairs} valid augmentation pairs (Original + Flip).[/dim]")

    if len(exact_duplicates) > 0:
        dup_table = Table(title="UNEXPECTED Identical Vectors Found (True Duplicates)", show_lines=True)
        dup_table.add_column("MD5 Hash", style="dim")
        dup_table.add_column("File Paths containing identical data", style="red")
        
        for h, paths in exact_duplicates.items():
            dup_table.add_row(h[:8] + "...", "\n".join(paths))
        console.print(dup_table)
    else:
        console.print("[bold green]No unexpected identical vectors found! Data is clean.[/bold green]")

    console.print("\n=== FINAL INTEGRITY REPORT ===")
    if total_data_leaks == 0 and len(exact_duplicates) == 0:
        console.print("[bold green]SUCCESS: The dataset is perfectly clean. Zero data leakage detected between First, Second, and Third runs.[/bold green]")
    else:
        console.print(f"[bold red]WARNING: Found {total_data_leaks} cross-session leaks and {len(exact_duplicates)} global duplicate groups![/bold red]")

if __name__ == "__main__":
    main()
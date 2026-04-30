import os
import shutil
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

console = Console()
FEATURES_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"

EXPECTED = {
    "walk": 6,
    "stairsup": 3,
    "stairsdown": 3,
    "slopeup": 6,
    "slopedown": 6
}

def get_canonical_action(action_name):
    a = action_name.lower().replace("_", "")
    if 'walk' in a: return "walk"
    if 'slopeup' in a: return "slopeup"
    if 'slopedown' in a: return "slopedown"
    if 'stairsup' in a or ('stairs' in a and 'up' in a): return "stairsup"
    if 'stairsdown' in a or ('stairs' in a and 'down' in a): return "stairsdown"
    return None

def check_dataset():
    console.print("[bold cyan]--- DATASET RUN COUNTER & FIXER ---[/bold cyan]")
    if not os.path.exists(FEATURES_ROOT):
        console.print("[red]Directory not found![/red]")
        return

    subjects = sorted([d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))])
    
    table = Table(title="Dataset Completeness Analysis", show_lines=True)
    table.add_column("Subject", style="cyan")
    table.add_column("Run", style="yellow")
    table.add_column("Walk (6)", justify="center")
    table.add_column("StUp (3)", justify="center")
    table.add_column("StDw (3)", justify="center")
    table.add_column("SlUp (6)", justify="center")
    table.add_column("SlDw (6)", justify="center")
    
    anomalies = []

    for subj in subjects:
        for run_name in ["FirstRun", "SecondRun", "ThirdRun"]:
            counts = {"walk": 0, "stairsup": 0, "stairsdown": 0, "slopeup": 0, "slopedown": 0}
            run_path = os.path.join(FEATURES_ROOT, subj, run_name)
            
            if not os.path.exists(run_path):
                continue
            
            for action in os.listdir(run_path):
                action_path = os.path.join(run_path, action)
                if not os.path.isdir(action_path) or 'debug' in action.lower():
                    continue
                    
                canon = get_canonical_action(action)
                if not canon:
                    continue
                    
                run_folders = [d for d in os.listdir(action_path) if d.startswith('Run_') and os.path.isdir(os.path.join(action_path, d))]
                
                counts[canon] += len(run_folders)
                
                if len(run_folders) > EXPECTED[canon]:
                    anomalies.append({
                        "subject": subj,
                        "run_name": run_name,
                        "action": action,
                        "path": action_path,
                        "found": len(run_folders),
                        "expected": EXPECTED[canon],
                        "folders": run_folders
                    })
            
            has_data = any(v > 0 for v in counts.values())
            if has_data:
                strs = {}
                for k, v in counts.items():
                    strs[k] = f"[green]{v}[/green]" if v == EXPECTED[k] else (f"[red]{v}[/red]" if v > 0 else "0")
                table.add_row(subj, run_name, strs["walk"], strs["stairsup"], strs["stairsdown"], strs["slopeup"], strs["slopedown"])

    console.print(table)
    
    if anomalies:
        console.print(f"\n[bold red]WARNING: Found {len(anomalies)} anomalies (excess runs)![/bold red]")
        for a in anomalies:
            console.print(f" - {a['subject']} | {a['run_name']} | {a['action']} -> Found {a['found']} (Expected {a['expected']})")
            
        if Confirm.ask("\nDo you want to MOVE the excess folders to a '_backup_extra_runs' directory, keeping only the FIRST acquisitions active?"):
            for a in anomalies:
                folders = a["folders"]
                folders.sort(key=lambda x: int(x.split('_')[1]))
                
                to_backup = folders[a["expected"]:]
                
                if not to_backup:
                    continue
                    
                backup_dir = os.path.join(a["path"], "_backup_extra_runs")
                os.makedirs(backup_dir, exist_ok=True)
                
                for d in to_backup:
                    src_path = os.path.abspath(os.path.join(a["path"], d))
                    dst_path = os.path.abspath(os.path.join(backup_dir, d))
                    
                    if os.path.exists(src_path):
                        try:
                            shutil.move(src_path, dst_path)
                            console.print(f"[green]Moved to backup:[/green] {d} -> _backup_extra_runs")
                        except Exception as e:
                            console.print(f"[red]Error moving {d}: {e}[/red]")
                    else:
                        console.print(f"[yellow]Directory not found (already moved?):[/yellow] {src_path}")
                        
            console.print("\n[bold green]Cleanup completed successfully! Dataset aligned and extra runs safely backed up.[/bold green]")
        else:
            console.print("[yellow]Operation cancelled. No files were moved.[/yellow]")
    else:
        console.print("\n[bold green]The dataset is perfectly aligned! No excess runs found.[/bold green]")

if __name__ == "__main__":
    check_dataset()
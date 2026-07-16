import os
import zipfile
import shutil
import re
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

SOURCE_IMU_ROOT = "/Volumes/LaCie/GAIT/IMUs" 
DEST_DATASET_ROOT = "/Volumes/LaCie/GAIT 2/dataset" 

IGNORE_NAMES = ["Export", "Output", "Session", "Mvn", "System", "Log", "__MACOSX"]

ACTION_MAP = {
    "slope_up": "slope_up", "stairs_up": "stairs_up", "strairs_up": "stairs_up",
    "slope_down": "slope_down", "stairs_down": "stairs_down", "strairs_down": "stairs_down",
    "walk": "walk"
}

SKIPPED_PEOPLE_LOG = set()

def resolve_target_folder(zip_subject_raw, existing_subjects):
    raw_subject = zip_subject_raw.strip()
    
    if raw_subject in IGNORE_NAMES or raw_subject.startswith("._") or "__MACOSX" in raw_subject: 
        return None, None, "SYSTEM_FOLDER"

    run_name = "FirstRun"
    base_name = raw_subject
    
    match_3 = re.search(r'[_ ]?3$', raw_subject)
    if match_3:
        run_name = "ThirdRun"
        base_name = raw_subject[:match_3.start()]
    else:
        match_2 = re.search(r'[_ ]?2$', raw_subject)
        if match_2:
            run_name = "SecondRun"
            base_name = raw_subject[:match_2.start()]

    base_clean = base_name.lower().replace(" ", "").replace("_", "")
    
    base_clean = base_clean.replace("arthur", "artur")
    
    if "alessio" in base_clean:
        if "f" in base_clean.replace("alessio", ""): 
            return "Alessio_F", run_name, "OK"
        return "Alessio", run_name, "OK"

    for subj in existing_subjects:
        if subj in ["Alessio", "Alessio_F"]: 
            continue
            
        subj_tokens = [t.lower() for t in re.split(r'[ _]+', subj) if t]
        if all(token in base_clean for token in subj_tokens):
            return subj, run_name, "OK"
            
        subj_clean = subj.lower().replace(" ", "").replace("_", "")
        if len(subj_clean) >= 6 and subj_clean[:6] in base_clean:
            return subj, run_name, "OK"

    return None, run_name, "NO_MATCH"

def process_imu_zip(zip_path, existing_subjects, progress):
    zip_name = os.path.basename(zip_path)
    debug_seen_folders = set()
    valid_csv_count = 0 

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_info in z.infolist():
                filename_clean = file_info.filename.replace('\\', '/')
                
                if file_info.is_dir() or filename_clean.startswith('._') or '/._' in filename_clean or '__MACOSX' in filename_clean:
                    continue

                if not filename_clean.lower().endswith('.csv'):
                    continue

                parts = filename_clean.split('/')
                
                if len(parts) < 3: 
                    if valid_csv_count == 0:
                        progress.console.print(f"[yellow]⚠️ Anomalous zip structure in {zip_name}: too few levels ({filename_clean})[/yellow]")
                    continue

                valid_csv_count += 1

                csv_filename = parts[-1]
                action_raw = parts[-2]
                
                zip_subject_raw = None
                for i in range(3, len(parts) + 1):
                    if parts[-i] not in IGNORE_NAMES:
                        zip_subject_raw = parts[-i]
                        break
                        
                if not zip_subject_raw:
                    continue
                
                debug_key = f"{zip_subject_raw}/{action_raw}"
                show_debug = False
                if debug_key not in debug_seen_folders:
                    debug_seen_folders.add(debug_key)
                    show_debug = True

                clean_subject, run_name, status = resolve_target_folder(zip_subject_raw, existing_subjects)
                
                if not clean_subject:
                    if status == "NO_MATCH" and show_debug:
                        progress.console.print(f"[red]SKIP (Unknown Subject):[/red] '{zip_subject_raw}' (in {zip_name}) -> No match in dataset.")
                        SKIPPED_PEOPLE_LOG.add(f"{zip_name} -> [{zip_subject_raw}]")
                    continue

                subject_base_dir = os.path.join(DEST_DATASET_ROOT, clean_subject)

                match = re.search(r"([A-Za-z_ ]+?)\s*-\s*0*(\d+)$", action_raw)
                if not match:
                    if show_debug:
                        progress.console.print(f"[yellow]SKIP (Wrong Action):[/yellow] '{action_raw}' (of {zip_subject_raw}) -> Folder does not end with '-001', '-002' etc.")
                    continue
                    
                base_action = match.group(1).replace(" ", "_").lower()
                run_num = match.group(2)

                canonical_action = ACTION_MAP.get(base_action, base_action)

                subject_dest_dir = os.path.join(subject_base_dir, run_name, "imu")
                dest_action_dir = os.path.join(subject_dest_dir, canonical_action, f"Run_{run_num}")
                dest_csv_path = os.path.join(dest_action_dir, csv_filename)
                
                if show_debug:
                    progress.console.print(f"[green]OK:[/green] '{zip_subject_raw}/{action_raw}' -> [cyan]{clean_subject}/{run_name}/imu/{canonical_action}/Run_{run_num}[/cyan]")

                if os.path.exists(dest_csv_path):
                    if show_debug:
                        progress.console.print(f"[dim]   -> Data already present, skip.[/dim]")
                    continue

                os.makedirs(dest_action_dir, exist_ok=True)
                
                with z.open(file_info) as source_file, open(dest_csv_path, "wb") as target_file:
                    shutil.copyfileobj(source_file, target_file)

        if valid_csv_count == 0:
            progress.console.print(f"[bold red]No compatible CSV files found in {zip_name}.[/bold red]")

    except Exception as e:
        progress.console.print(f"[bold red]ERROR ZIP {zip_name}: {e}[/bold red]")

def main():
    console.print(Panel.fit("[bold gradient(cyan,magenta)]--- IMU Organizer ---[/bold gradient(cyan,magenta)]"))
    
    if not os.path.exists(DEST_DATASET_ROOT):
        console.print(f"[red]Error: Destination folder not found: {DEST_DATASET_ROOT}[/red]")
        return

    existing_video_subjects = [d for d in os.listdir(DEST_DATASET_ROOT) if os.path.isdir(os.path.join(DEST_DATASET_ROOT, d)) and d not in IGNORE_NAMES]
    console.print(f"Subjects in dataset (Gallery): [green]{len(existing_video_subjects)}[/green]")

    zip_files = sorted([os.path.join(SOURCE_IMU_ROOT, f) for f in os.listdir(SOURCE_IMU_ROOT) if f.endswith(".zip") and not f.startswith("._")])
    
    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as progress:
        task = progress.add_task("[cyan]Analyzing ZIP files and Extracting IMU data...", total=len(zip_files))
        for zip_file in zip_files:
            progress.console.print(f"\n[bold magenta]>>> Opening: {os.path.basename(zip_file)}[/bold magenta]")
            process_imu_zip(zip_file, existing_video_subjects, progress)
            progress.update(task, advance=1)
            
    console.print("\n[bold green]IMU Organization Completed Successfully![/bold green]")

if __name__ == "__main__":
    main()
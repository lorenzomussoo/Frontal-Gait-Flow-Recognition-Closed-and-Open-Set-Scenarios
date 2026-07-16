import os
import re
import zipfile
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()

SOURCE_VIDEO_ROOTS = {
    "FirstRun": "/Volumes/LaCie/GAIT/FirstRun/RGBD",
    "SecondRun": "/Volumes/LaCie/GAIT 2/SecondRun/RGBD",
    "ThirdRun": "/Volumes/LaCie/GAIT 2/ThirdRun/RGBD"
}
SOURCE_IMU_ROOT = "/Volumes/LaCie/GAIT/IMUs"
DEST_DATASET_ROOT = "/Volumes/LaCie/GAIT 2/dataset" 

EXPECTED_RUNS = ["FirstRun", "SecondRun", "ThirdRun"]
IGNORE_NAMES = ["Export", "Output", "Session", "Mvn", "System", "Log", "__MACOSX"]

ACTION_MAP = {
    "Slope_up": "slope_up", "Stairs_up": "stairs_up", "Up": "stairs_up", "SlopeUp": "slope_up", "StairsUp": "stairs_up",
    "Slope_down": "slope_down", "Stairs_down": "stairs_down", "Down": "stairs_down", "SlopeDown": "slope_down", "StairsDown": "stairs_down",
    "Walk": "walk", "walk": "walk",
    "slope_up": "slope_up", "stairs_up": "stairs_up", "strairs_up": "stairs_up",
    "slope_down": "slope_down", "stairs_down": "stairs_down", "strairs_down": "stairs_down",
}

SUBJECT_MAP = {
    "Alessio F": "Alessio_F",
    "Alessio_F": "Alessio_F",
    "AndreaPrincic_2": "Andrea",
    "Artur_2": "Artur",
    "Camilla 2": "Camilla",
    "Chiara2": "Chiara",
    "Diego_2": "Diego",
    "Eduardo_2": "Eduardo",
    "Eleonora 2": "Eleonora",
    "Federico2": "Federico",
    "Laura_2": "Laura",
    "Manuel_Gil_2": "Manuel",
    "MariaVittroria2": "MariaVittoria", 
    "Romeo_2": "Romeo",
    "ValerioVenanzi_2": "Valerio",
    "Vito2": "Vito",
    "AlessioFerrone_3": "Alessio_F",
    "Andrea P 3": "Andrea",
    "Artur_3": "Artur",
    "Camilla3": "Camilla",
    "Chiara_3": "Chiara",
    "Diego_3": "Diego",
    "Eduardo_3": "Eduardo",
    "Eleonora3": "Eleonora",
    "Federico_Fontana_3": "Federico",
    "Francesco_3": "Francesco",
    "JessicaFrabotta_3": "Jessica",
    "Lorenzo 3": "Lorenzo",
    "Manuel_3": "Manuel",
    "Matteo_Basile_3": "Matteo",
    "Romeo 3": "Romeo",
    "Valerio_3": "Valerio"
}

expected_video = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
expected_imu = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
found_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))) 

def normalize_subject_name(name):
    return SUBJECT_MAP.get(name, name.replace(" ", "_").strip())

def resolve_imu_run_and_subject(zip_subject_raw, existing_subjects):
    raw_subject = zip_subject_raw.strip()
    
    if raw_subject in IGNORE_NAMES or raw_subject.startswith("._") or "__MACOSX" in raw_subject: 
        return None, None

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
            return "Alessio_F", run_name
        return "Alessio", run_name

    for subj in existing_subjects:
        if subj in ["Alessio", "Alessio_F"]: 
            continue
            
        subj_tokens = [t.lower() for t in re.split(r'[ _]+', subj) if t]
        if all(token in base_clean for token in subj_tokens):
            return subj, run_name
            
        subj_clean = subj.lower().replace(" ", "").replace("_", "")
        if len(subj_clean) >= 6 and subj_clean[:6] in base_clean:
            return subj, run_name

    return None, run_name

def scan_source_video():
    console.print("[yellow]1. Scanning Video source (.bag)...[/yellow]")
    for run_name, source_root in SOURCE_VIDEO_ROOTS.items():
        if not os.path.exists(source_root): continue
        
        for root, dirs, files in os.walk(source_root):
            for f in files:
                if f.endswith('.bag') and not f.startswith('._'):
                    parts = root.split(os.sep)
                    if len(parts) < 2: continue
                    
                    action_raw = parts[-1]
                    subject_raw = parts[-2]
                    
                    subject = normalize_subject_name(subject_raw)
                    action_std = ACTION_MAP.get(action_raw, action_raw.lower())
                    
                    expected_video[subject][run_name][action_std] += 1

def scan_source_imu():
    console.print("[yellow]2. Scanning IMU source (.zip)...[/yellow]")
    known_video = set(expected_video.keys())
    if os.path.exists(DEST_DATASET_ROOT):
        known_video.update([d for d in os.listdir(DEST_DATASET_ROOT) if os.path.isdir(os.path.join(DEST_DATASET_ROOT, d)) and d not in IGNORE_NAMES])
        
    zip_files = sorted([os.path.join(SOURCE_IMU_ROOT, f) for f in os.listdir(SOURCE_IMU_ROOT) if f.endswith('.zip') and not f.startswith('._')])
    
    scan_source_imu.unique_runs = set()
    
    for z_path in track(zip_files, description="Reading Zip Indexes..."):
        try:
            with zipfile.ZipFile(z_path, 'r') as zf:
                for file_info in zf.infolist():
                    filename_clean = file_info.filename.replace('\\', '/')
                    if file_info.is_dir() or filename_clean.startswith('._') or '/._' in filename_clean or '__MACOSX' in filename_clean:
                        continue
                    if not filename_clean.lower().endswith('.csv'): continue
                    
                    parts = filename_clean.split('/')
                    if len(parts) < 3: continue
                    
                    action_raw = parts[-2]
                    zip_subject_raw = None
                    for i in range(3, len(parts) + 1):
                        if parts[-i] not in IGNORE_NAMES:
                            zip_subject_raw = parts[-i]
                            break
                    
                    if not zip_subject_raw: continue
                    
                    tgt_subj, run_name = resolve_imu_run_and_subject(zip_subject_raw, known_video)
                    if not tgt_subj: continue
                    
                    match = re.search(r"([A-Za-z_ ]+?)\s*-\s*0*(\d+)$", action_raw)
                    if not match: continue
                    
                    base_action = match.group(1).replace(" ", "_").lower()
                    act_std = ACTION_MAP.get(base_action, base_action)
                    run_num = match.group(2)
                    
                    key = f"{tgt_subj}|{run_name}|{act_std}|{run_num}"
                    if key not in scan_source_imu.unique_runs:
                        expected_imu[tgt_subj][run_name][act_std] += 1
                        scan_source_imu.unique_runs.add(key)
        except: pass

def scan_destination():
    console.print("[yellow]3. Scanning Final Dataset...[/yellow]")
    if not os.path.exists(DEST_DATASET_ROOT): return

    for subj in os.listdir(DEST_DATASET_ROOT):
        subj_path = os.path.join(DEST_DATASET_ROOT, subj)
        if not os.path.isdir(subj_path) or subj in IGNORE_NAMES: continue
        
        for run_name in EXPECTED_RUNS:
            run_dir = os.path.join(subj_path, run_name)
            if not os.path.isdir(run_dir): continue
            
            for mod in ['depth', 'rgb', 'ir']:
                mod_path = os.path.join(run_dir, mod)
                if os.path.exists(mod_path):
                    for act in os.listdir(mod_path):
                        if act.startswith('.'): continue
                        act_std = ACTION_MAP.get(act, act) 
                        act_path = os.path.join(mod_path, act)
                        if os.path.isdir(act_path):
                            cnt = len([f for f in os.listdir(act_path) if f.endswith('.avi') and not f.startswith('._')])
                            found_data[subj][run_name][act_std][mod] += cnt 
            
            imu_path = os.path.join(run_dir, 'imu')
            if os.path.exists(imu_path):
                for act in os.listdir(imu_path):
                    if act.startswith('.'): continue
                    act_std = ACTION_MAP.get(act, act)
                    act_path = os.path.join(imu_path, act)
                    if os.path.isdir(act_path):
                        cnt = len([d for d in os.listdir(act_path) if d.startswith('Run_')])
                        found_data[subj][run_name][act_std]['imu'] += cnt

def generate_report():
    console.print(Panel.fit("[bold gradient(cyan,magenta)]--- DATASET AUDIT ---[/bold gradient(cyan,magenta)]"))
    
    all_subjs = sorted(list(set(expected_video.keys()) | set(found_data.keys()) | set(expected_imu.keys())))
    ok_count = 0
    err_count = 0
    
    for subj in all_subjs:
        subj_issues = []
        
        for run_name in EXPECTED_RUNS:
            run_issues = []
            actions = sorted(list(
                set(expected_video[subj][run_name].keys()) | 
                set(found_data[subj][run_name].keys()) |
                set(expected_imu[subj][run_name].keys())
            ))
            
            for act in actions:
                exp_v = expected_video[subj][run_name][act]
                exp_i = expected_imu[subj][run_name][act]
                
                got_d = found_data[subj][run_name][act]['depth']
                got_r = found_data[subj][run_name][act]['rgb']
                got_i = found_data[subj][run_name][act]['ir']
                got_imu = found_data[subj][run_name][act]['imu']
                
                if exp_v > 0:
                    if got_d != exp_v: run_issues.append(f"[{act}] Depth: Expected {exp_v}, Found {got_d}")
                    if got_r != exp_v: run_issues.append(f"[{act}] RGB: Expected {exp_v}, Found {got_r}")
                    if got_i != exp_v: run_issues.append(f"[{act}] IR: Expected {exp_v}, Found {got_i}")
                
                if exp_i > 0 and got_imu != exp_i:
                    run_issues.append(f"[{act}] IMU: Expected {exp_i} runs, Found {got_imu}")
            
            if run_issues:
                subj_issues.append((run_name, run_issues))
        
        if subj_issues:
            err_count += 1
            console.print(f"[bold red]❌ {subj}[/bold red]")
            for r_name, r_iss in subj_issues:
                console.print(f"   [yellow]{r_name}[/yellow]:")
                for i in r_iss: console.print(f"      -> {i}")
        else:
            ok_count += 1
            
    console.print(f"\n[green]Subjects OK: {ok_count}[/green] | [red]Subjects with Errors: {err_count}[/red]")
    if err_count == 0:
        console.print(Panel.fit("[bold green]PERFECT DATASET![/bold green]\nAll folders are standardized and complete across all runs."))

if __name__ == "__main__":
    scan_source_video()
    scan_source_imu()
    scan_destination()
    generate_report()
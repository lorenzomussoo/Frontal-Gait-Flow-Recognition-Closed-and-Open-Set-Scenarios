import os
import shutil
from rich.console import Console
from rich.panel import Panel

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT 2/dataset" 

def migrate_first_run():
    console.print(Panel.fit("[bold cyan]--- Migrazione Struttura FirstRun ---[/bold cyan]"))
    
    if not os.path.exists(DATASET_ROOT):
        console.print(f"[red]Errore: Cartella dataset non trovata in {DATASET_ROOT}[/red]")
        return

    subjects = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    
    count = 0
    
    for subject in subjects:
        if subject in ["Export", "System", "__MACOSX"]:
            continue
            
        subj_path = os.path.join(DATASET_ROOT, subject)
        first_run_path = os.path.join(subj_path, "FirstRun")
        
        folders_to_move = ["depth", "rgb", "ir"]
        moved_something = False
        
        for folder_name in folders_to_move:
            src_folder = os.path.join(subj_path, folder_name)
            
            if os.path.exists(src_folder):
                if not os.path.exists(first_run_path):
                    os.makedirs(first_run_path)
                
                dst_folder = os.path.join(first_run_path, folder_name)
                if os.path.exists(dst_folder):
                    console.print(f"[dim]Attenzione: {dst_folder} esiste già. Skippo.[/dim]")
                    continue
                    
                shutil.move(src_folder, dst_folder)
                moved_something = True
        
        if moved_something:
            console.print(f"[green]Aggiornato soggetto:[/green] [yellow]{subject}[/yellow]")
            count += 1
            
    console.print(f"\n[bold green]Finito! Sono stati migrati {count} soggetti.[/bold green]")
    console.print("[dim]La cartella 'imu' è rimasta intatta.[/dim]")

if __name__ == "__main__":
    migrate_first_run()
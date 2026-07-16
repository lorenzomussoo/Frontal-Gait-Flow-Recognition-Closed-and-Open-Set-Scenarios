import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

DATASET_ROOT = "/Volumes/LaCie/GAIT 2/dataset"

EXPECTED_RUNS = ["FirstRun", "SecondRun", "ThirdRun"]
ACTIONS = ["walk", "stairs_up", "stairs_down", "slope_up", "slope_down"]

def audit_imu():
    console.print(Panel.fit("[bold gradient(cyan,magenta)]--- IMU Dataset Audit ---[/bold gradient(cyan,magenta)]"))

    if not os.path.exists(DATASET_ROOT):
        console.print(f"[red]Error: Dataset not found in {DATASET_ROOT}[/red]")
        return

    subjects = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d not in ["Export", "System", "__MACOSX"]])

    table = Table(title="IMU Acquisition Status", show_header=True, header_style="bold magenta")
    table.add_column("Subject", style="cyan", no_wrap=True)
    table.add_column("Session", style="yellow")
    table.add_column("Walk", justify="center")
    table.add_column("Stairs Up", justify="center")
    table.add_column("Stairs Down", justify="center")
    table.add_column("Slope Up", justify="center")
    table.add_column("Slope Down", justify="center")
    table.add_column("Note / Warnings", style="red")

    summary_issues = {s: {"missing_runs": [], "incomplete": []} for s in subjects}

    for subject in subjects:
        subj_dir = os.path.join(DATASET_ROOT, subject)
        
        for run_name in EXPECTED_RUNS:
            run_dir = os.path.join(subj_dir, run_name)
            
            if not os.path.exists(run_dir):
                summary_issues[subject]["missing_runs"].append(run_name)
                continue
            
            imu_dir = os.path.join(run_dir, "imu")
            
            if not os.path.exists(imu_dir):
                table.add_row(subject, run_name, "-", "-", "-", "-", "-", "❌ No IMU Folder")
                summary_issues[subject]["incomplete"].append(f"{run_name} (No IMU folder)")
                continue
            
            counts = {}
            notes = []
            
            for action in ACTIONS:
                action_dir = os.path.join(imu_dir, action)
                if os.path.exists(action_dir):
                    runs_found = [d for d in os.listdir(action_dir) if d.startswith("Run_")]
                    counts[action] = len(runs_found)
                else:
                    counts[action] = 0

            walk_expected = 6
            
            w_c = counts["walk"]
            su_c = counts["stairs_up"]
            sd_c = counts["stairs_down"]
            slu_c = counts["slope_up"]
            sld_c = counts["slope_down"]

            if w_c == 0: notes.append("No Walk")
            elif w_c < walk_expected: notes.append(f"Walk Incomplete ({w_c}/{walk_expected})")
            elif w_c > walk_expected: notes.append(f"Walk Extra ({w_c})")
            
            if su_c == 0 and sd_c == 0: notes.append("No Scale")
            elif su_c < 3 or sd_c < 3: notes.append("Scale Incomplete")

            if slu_c == 0 and sld_c == 0: notes.append("No Slopes")
            elif slu_c < 3 or sld_c < 3: notes.append("Slopes Incomplete")

            def fmt(count, expected=3):
                if count == 0: return "[red]0[/red]"
                elif count < expected: return f"[yellow]{count}[/yellow]"
                else: return f"[green]{count}[/green]"

            w_str = fmt(w_c, walk_expected)
            su_str = fmt(su_c, 3)
            sd_str = fmt(sd_c, 3)
            slu_str = fmt(slu_c, 3)
            sld_str = fmt(sld_c, 3)

            note_str = ", ".join(notes) if notes else "[green] Complete[/green]"

            table.add_row(subject, run_name, w_str, su_str, sd_str, slu_str, sld_str, note_str)

            if notes:
                summary_issues[subject]["incomplete"].append(f"{run_name} ({', '.join(notes)})")

    console.print(table)
    console.print("\n[dim]Legend: [green]Green (Complete)[/green], [yellow]Yellow (Incomplete)[/yellow], [red]Red (Missing)[/red][/dim]\n")

    console.print(Panel("[bold yellow]Incomplete or Missing Runs Summary[/bold yellow]"))
    for subj, issues in summary_issues.items():
        if issues["missing_runs"] or issues["incomplete"]:
            console.print(f"[bold cyan]{subj}[/bold cyan]:")
            if issues["missing_runs"]:
                console.print(f"  [red]Missing Runs:[/red] {', '.join(issues['missing_runs'])}")
            if issues["incomplete"]:
                console.print(f"  [yellow]Incomplete:[/yellow] {'; '.join(issues['incomplete'])}")
            console.print("")

if __name__ == "__main__":
    audit_imu()
import os
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

PROCESSED_ROOT = "/Volumes/LaCie/GAIT 2/processed_features"
DATASET_ROOT = "/Volumes/LaCie/GAIT 2/dataset"

IMU_FILES = [
    "Sensor_Free_Acceleration.csv", "Sensor_Orientation_Euler.csv", "Sensor_Magnetic_Field.csv", "Sensor_Orientation_Quat.csv",
    "Segment_Velocity.csv", "Segment_Angular_Velocity.csv", "Segment_Position.csv", "Segment_Orientation_Euler.csv", "Segment_Orientation_Quat.csv",
    "Segment_Acceleration.csv", "Segment_Angular_Acceleration.csv",
    "Joint_Angles_ZXY.csv", "Joint_Angles_XZY.csv", "Ergonomic_Joint_Angles_ZXY.csv", "Ergonomic_Joint_Angles_XZY.csv",
    "Center_of_Mass.csv", "Marker.csv", "Frame_Rate.csv", "TimeStamp.csv"
]

def check():
    console.print("[bold magenta]--- Dimension Checker ---[/bold magenta]")
    
    test_file = None
    if os.path.exists(PROCESSED_ROOT):
        for root, _, files in os.walk(PROCESSED_ROOT):
            for f in files:
                if f.endswith('.npy') and not f.endswith('flip.npy'):
                    test_file = os.path.join(root, f)
                    break
            if test_file: break
            
    if not test_file:
        console.print(f"[red]Error: No .npy file found in {PROCESSED_ROOT}[/red]")
        return

    console.print(f"[dim]Testing file:[/dim] {test_file}")
    vector = np.load(test_file)
    total_len = len(vector)
    console.print(f"[bold]Total Vector Length:[/bold] {total_len}")
    
    console.print("[dim]Computing IMU dimension from original CSV files...[/dim]")
    
    schema = {}
    for root, _, files in os.walk(DATASET_ROOT):
        for needed_file in IMU_FILES:
            if needed_file not in schema and needed_file in files:
                try:
                    path = os.path.join(root, needed_file)
                    df = pd.read_csv(path, sep=';', nrows=1)
                    if df.shape[1] < 2: df = pd.read_csv(path, sep=',', nrows=1)
                    cols = [c for c in df.columns if "Frame" not in c and "Time" not in c]
                    schema[needed_file] = len(cols)
                except: pass
        if len(schema) == len(IMU_FILES): break
        
    imu_len = sum([c * 5 for c in schema.values()])
            
    console.print(f"[bold]IMU Length:[/bold] {imu_len}")
    
    video_len_total = total_len - imu_len
    single_modality_len = video_len_total // 2
    
    console.print(f"\n[bold green]RESULT:[/bold green]")
    console.print(f"Total ({total_len}) - IMU ({imu_len}) = [bold cyan]{video_len_total}[/bold cyan] (Depth + RGB combined)")
    console.print(f"Dimension for single camera: {video_len_total} / 2 = [bold magenta]{single_modality_len}[/bold magenta]")
    
    if single_modality_len == 24576:
        console.print("\n[bold green]CONFIRMED: GLOBAL_VIDEO_LEN = 24576 is CORRECT![/bold green]")
    else:
        console.print(f"\n[bold red]ATTENTION: The correct GLOBAL_VIDEO_LEN should be {single_modality_len}[/bold red]")

if __name__ == "__main__":
    check()
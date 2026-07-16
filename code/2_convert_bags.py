import pyrealsense2 as rs
import numpy as np
import cv2
import os
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

DEST_ROOT = "/Volumes/LaCie/GAIT 2/dataset" 
RUNS_TO_PROCESS = [
    {"name": "FirstRun",  "source": "/Volumes/LaCie/GAIT/FirstRun/RGBD"},
    {"name": "SecondRun", "source": "/Volumes/LaCie/GAIT 2/SecondRun/RGBD"},
    {"name": "ThirdRun",  "source": "/Volumes/LaCie/GAIT 2/ThirdRun/RGBD"}
]

CLIPPING_DISTANCE_METERS = 15.0  
DEPTH_SCALE = 0.001             

SUBJECT_MAP = {
    "Alessio": "Alessio",
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
    "Francesco": "Francesco",
    "Jessica": "Jessica",
    "Laura_2": "Laura",
    "Lorenzo": "Lorenzo",
    "Luca": "Luca",
    "Manuel_Gil_2": "Manuel",
    "Marco": "Marco",
    "MariaVittroria2": "MariaVittoria", 
    "Matteo": "Matteo",
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

def get_video_info_from_path(bag_path):
    parts = bag_path.split(os.sep)
    action_raw = parts[-2]
    raw_subject = parts[-3]

    clean_subject = SUBJECT_MAP.get(raw_subject, raw_subject.replace(" ", "_"))
    
    return clean_subject, action_raw

def normalize_action_name(raw_action):
    mapping = {
        "Walk": "walk",
        "StairsUp": "stairs_up", "Up": "stairs_up",
        "StairsDown": "stairs_down", "Down": "stairs_down"
    }
    return mapping.get(raw_action, raw_action)

def setup_video_writer(dest_path, width, height, fps, is_color):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(dest_path, fourcc, fps, (width, height), isColor=is_color)

def convert_bag_to_multimodal(bag_path, subject_dest_dir, action_folder_name, filename_base, run_name):
    run_dir = os.path.join(subject_dest_dir, run_name)

    dir_depth = os.path.join(run_dir, "depth", action_folder_name)
    dir_rgb = os.path.join(run_dir, "rgb", action_folder_name)
    dir_ir = os.path.join(run_dir, "ir", action_folder_name)
    
    os.makedirs(dir_depth, exist_ok=True)
    os.makedirs(dir_rgb, exist_ok=True)
    os.makedirs(dir_ir, exist_ok=True)

    path_depth = os.path.join(dir_depth, f"{filename_base}.avi")
    path_rgb = os.path.join(dir_rgb, f"{filename_base}.avi")
    path_ir = os.path.join(dir_ir, f"{filename_base}.avi")

    if os.path.exists(path_depth) and os.path.exists(path_rgb) and os.path.exists(path_ir):
        return True, "SKIPPED (Exists)"

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        return False, f"Corrupt BAG/Driver: {e}"

    try:
        stream_d = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        out_depth = setup_video_writer(path_depth, stream_d.width(), stream_d.height(), stream_d.fps(), False)

        stream_c = profile.get_stream(rs.stream.color).as_video_stream_profile()
        out_rgb = setup_video_writer(path_rgb, stream_c.width(), stream_c.height(), stream_c.fps(), True)

        stream_ir = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        out_ir = setup_video_writer(path_ir, stream_ir.width(), stream_ir.height(), stream_ir.fps(), False)
    except Exception as e:
        return False, f"Stream Missing: {e}"

    clipping_dist_raw = CLIPPING_DISTANCE_METERS / DEPTH_SCALE

    try:
        while True:
            try:
                success, frames = pipeline.try_wait_for_frames(timeout_ms=1000)
                if not success: break 
            except RuntimeError:
                break
            
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                mask = np.logical_and(depth_image > 0, depth_image < clipping_dist_raw)
                norm_image = np.zeros_like(depth_image, dtype=np.uint8)
                if np.any(mask):
                    inverted_depth = clipping_dist_raw - depth_image[mask]
                    norm_image[mask] = (inverted_depth * 255 / clipping_dist_raw).astype(np.uint8)
                out_depth.write(norm_image)

            color_frame = frames.get_color_frame()
            if color_frame:
                out_rgb.write(cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR))
            ir_frame = frames.get_infrared_frame(1)
            if ir_frame:
                out_ir.write(np.asanyarray(ir_frame.get_data()))

    finally:
        pipeline.stop()
        if 'out_depth' in locals(): out_depth.release()
        if 'out_rgb' in locals(): out_rgb.release()
        if 'out_ir' in locals(): out_ir.release()

    return True, "OK"

def main():
    console.print(Panel.fit("[bold gradient(cyan,magenta)]--- Bag Converter ---[/bold gradient(cyan,magenta)]"))

    for run in RUNS_TO_PROCESS:
        run_name = run["name"]
        source_dir = run["source"]
        
        console.print(f"\n[bold yellow]>>> Processing {run_name}...[/bold yellow]")

        if not os.path.exists(source_dir):
            console.print(f"[dim]Folder not found: {source_dir}. Skipping this run.[/dim]")
            continue

        all_bag_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.startswith("._"): continue
                if file.endswith(".bag"):
                    all_bag_files.append(os.path.join(root, file))
        all_bag_files.sort()
        
        console.print(f"Found [yellow]{len(all_bag_files)}[/yellow] files in {run_name}.")

        counters = {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {run_name}...", total=len(all_bag_files))
            
            for bag_path in all_bag_files:
                subject, raw_action = get_video_info_from_path(bag_path)
                clean_action = normalize_action_name(raw_action)
                
                counter_key = f"{subject}_{clean_action}"
                if counter_key not in counters: counters[counter_key] = 0
                counters[counter_key] += 1
                
                filename_base = f"{subject}_{clean_action}_{counters[counter_key]}"
                subject_dest_dir = os.path.join(DEST_ROOT, subject)
                
                ok, msg = convert_bag_to_multimodal(bag_path, subject_dest_dir, clean_action, filename_base, run_name)
                
                if not ok and "SKIPPED" not in msg:
                    progress.log(f"[red]Error on {filename_base}: {msg}[/red]")
                
                progress.update(task, advance=1)
                
    console.print("\n[bold green]Bag Conversion Completed![/bold green]")

if __name__ == "__main__":
    main()
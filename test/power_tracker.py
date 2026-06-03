import sys
import os
import cv2
import yaml
import torch
import json
import shutil
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# --- 1. SETUP PATHS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dynaphos import utils, cortex_models
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import Map
from dynaphos.image_processing import canny_processor, sobel_processor, scale_image

# --- HELPER FUNCTIONS ---

def get_center_crop(frame):
    """Crops the frame to a square center."""
    h, w = frame.shape[:2]
    s = min(h, w)
    start_y = (h - s) // 2
    start_x = (w - s) // 2
    crop = frame[start_y:start_y+s, start_x:start_x+s]
    return crop

def preprocess_frame(frame, target_res, method='none'):
    """Applies resizing and edge detection."""
    frame = cv2.resize(frame, target_res, interpolation=cv2.INTER_AREA)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    if method == 'canny':
        processed = canny_processor(frame, threshold_low=20, threshold_high=75)
    elif method == 'sobel':
        processed = sobel_processor(frame)
        processed = scale_image(processed, f=255.0, use_max=True).astype(np.uint8)
    else:
        processed = frame
        
    return processed

def save_run_metadata(output_dir, video_name, fps, total_frames, electrode_count, 
                      params, args, run_duration, seed):
    """Saves a JSON file with all context needed to reproduce/analyze the data."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "video_source": video_name,
        "video_fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
        "electrode_count": electrode_count,
        "processing_time_s": run_duration,
        "seed_used": seed,
        "settings": {
            "raster_mode": args.raster,
            "raster_groups": args.groups,
            "edge_detection": args.edge_detection,
            "resolution": params['run']['resolution']
        },
        "simulation_params": {
            "pulse_width": params['default_stim']['pw_default'],
            "frequency": params['default_stim']['freq_default']
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

def run_extraction():
    # --- ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Extract raw stimulation data from video.")
    parser.add_argument('--video', type=str, required=True, 
                        help="Path to input video (relative to project root or absolute)")
    
    parser.add_argument('--raster', type=str, default='none', 
                        choices=['none', 'checkerboard', 'random', 'horizontal', 'vertical'])
    parser.add_argument('--groups', type=int, default=5, help="Number of raster groups")
    
    parser.add_argument('--edge-detection', type=str, default='none',
                        choices=['none', 'canny', 'sobel'])
    
    parser.add_argument('--output-root', type=str, default='data_logs',
                        help="Root directory for saved data")
    
    args = parser.parse_args()

    # --- SETUP & VALIDATION ---
    video_path = Path(args.video)
    if not video_path.exists():
        potential_path = project_root / 'videos' / args.video
        if potential_path.exists():
            video_path = potential_path
        else:
            print(f"❌ Error: Video not found at {video_path}")
            return

    # Auto-calculate Raster Rate based on FPS
    temp_cap = cv2.VideoCapture(str(video_path))
    video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
    total_frames_est = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps == 0: video_fps = 20.0 
    temp_cap.release()
    
    if args.raster != 'none':
        calculated_rate = video_fps / args.groups
    else:
        calculated_rate = 0.0

    # --- ORGANIZED OUTPUT STRUCTURE ---
    video_stem = video_path.stem
    config_name = f"{args.edge_detection}_{args.raster}"
    if args.raster != 'none':
        config_name += f"_g{args.groups}"
        
    output_dir = project_root / args.output_root / video_stem / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Output Directory: {output_dir}")

    # --- LOAD PARAMS ---
    config_path = project_root / 'config'
    params_file = config_path / 'params.yaml'
    shutil.copy(params_file, output_dir / "params_snapshot.yaml")
    
    with open(params_file, 'r') as f: params = yaml.safe_load(f)

    # --- SEED IMPLEMENTATION ---
    # 1. Extract seed from params
    seed = params['run'].get('seed', 42)
    print(f"🌱 Using Simulation Seed: {seed}")
    
    # 2. Create Deterministic RNG
    rng = np.random.default_rng(seed)
    
    # 3. Set Global Seeds (for safety, though we pass rng explicitly where possible)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- DEVICE SETUP ---
    if torch.cuda.is_available():
        params['run']['gpu'] = 0
        device = 'cuda'
        print("✅ GPU Acceleration Enabled")
    else:
        params['run']['gpu'] = False
        device = 'cpu'

    # --- COORDINATE GENERATION ---
    grid_file = config_path / 'grid_coords_dipole_valid.yaml'
    if not grid_file.exists(): grid_file = config_path / 'grid_coords_dipole.yaml'
    
    x, y = utils.load_coordinates_from_yaml(str(grid_file))
    coordinates_cortex = Map(x=x, y=y)
    
    print("Generating phosphene map...")
    # 4. Pass the seeded RNG to the model
    # This ensures dropout and noise are consistent every time
    try:
        phosphene_coords = cortex_models.get_visual_field_coordinates_from_cortex_full(
            params['cortex_model'], coordinates_cortex, rng=rng
        )
        # Extract Visual Field Cartesian Coordinates
        # These correspond 1-to-1 with the surviving electrodes
        visual_x = phosphene_coords._x
        visual_y = phosphene_coords._y
    except AttributeError:
        # Fallback for different library versions
        phosphene_coords = cortex_models.get_visual_field_coordinates_from_cortex(
            params['cortex_model'], coordinates_cortex, rng=rng
        )
        if hasattr(phosphene_coords, 'x'):
            visual_x, visual_y = phosphene_coords.x, phosphene_coords.y
        else:
            visual_x, visual_y = phosphene_coords._x, phosphene_coords._y

    n_electrodes = len(visual_x)
    print(f"ℹ️  Functional Electrodes: {n_electrodes}")

    # --- SAVE COORDINATES ---
    # Save the coordinates of the surviving electrodes immediately
    print(f"💾 Saving coordinates for {n_electrodes} electrodes...")
    coords_array = np.column_stack((visual_x, visual_y))
    np.save(output_dir / "coords_visual.npy", coords_array)

    # --- INITIALIZE SIMULATOR ---
    # Note: We pass the SAME seed implicitly via params['run']['seed'] if the simulator uses it internally,
    # but we also pass our 'phosphene_coords' which are already deterministic.
    sim = GaussianSimulator(
        params, 
        phosphene_coords,
        rng=rng,  # Pass the same RNG to simulator for threshold generation
        raster_enabled=(args.raster != 'none'),
        raster_pattern=args.raster,
        raster_num_groups=args.groups,
        raster_rate_hz=calculated_rate
    )

    # --- SAVE STATIC DATA (IMPEDANCE) ---
    impedances = sim.impedance.get().cpu().numpy()
    np.save(output_dir / "impedances.npy", impedances)

    # --- EXTRACTION LOOP ---
    cap = cv2.VideoCapture(str(video_path))
    dt = 1.0 / video_fps
    target_res = tuple(params['run']['resolution'])
    
    amplitude_history = []
    
    print(f"🚀 Starting Extraction: {total_frames_est} frames expected.")
    start_time = datetime.now()
    frame_nr = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break 
            frame_nr += 1
            
            # Preprocess & Stimulate
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = get_center_crop(gray)
            processed_img = preprocess_frame(crop, target_res, method=args.edge_detection)
            img_norm = processed_img.astype(np.float32) / 255.0
            
            raw_stim_amp = sim.sample_stimulus(img_norm, rescale=True)
            sim.update(raw_stim_amp, dt=dt)
            
            # Masking
            current_mask = sim.get_current_raster_mask()
            delivered_amplitude = raw_stim_amp.view(sim.shape) * current_mask
            
            amplitude_history.append(delivered_amplitude.detach().cpu().numpy().flatten())
            
            if frame_nr % 100 == 0:
                print(f"   Processed {frame_nr} frames...", end='\r')
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Saving partial data...")
    finally:
        cap.release()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✅ Extraction Complete in {duration:.2f}s")

    # --- SAVE TIME-SERIES DATA ---
    print("💾 Saving Amplitude history...")
    full_amplitude_array = np.array(amplitude_history, dtype=np.float32)
    np.save(output_dir / "amplitudes.npy", full_amplitude_array)
    
    # Save Metadata with Seed
    save_run_metadata(output_dir, video_path.name, video_fps, frame_nr, n_electrodes, 
                      params, args, duration, seed)

    print(f"🎉 All data saved to: {output_dir}")

if __name__ == "__main__":
    run_extraction()
import sys
import os
import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. SETUP PATHS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import Map
from dynaphos import cortex_models 
from dynaphos.utils import cartesian_to_complex

def map_coords_to_pixels(x_coords, y_coords, img_size=(800, 800), padding=50):
    """Linearly map physical coordinates to image pixel coordinates."""
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x == 0: range_x = 1
    if range_y == 0: range_y = 1
    
    draw_w = img_size[0] - 2 * padding
    draw_h = img_size[1] - 2 * padding
    
    px = (padding + (x_coords - min_x) / range_x * draw_w).astype(int)
    py = (padding + (draw_h - (y_coords - min_y) / range_y * draw_h)).astype(int)
    return px, py

def get_center_crop(frame, target_size=None):
    """Returns a center cropped square of the frame."""
    h, w = frame.shape[:2]
    s = min(h, w)
    start_y = (h - s) // 2
    start_x = (w - s) // 2
    crop = frame[start_y:start_y+s, start_x:start_x+s]
    if target_size:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return crop

def run_video_tracking():
    # --- 2. CONFIGURATION ---
    video_filename = "aria/loc3_script4_seq2_rec1/povvideocut.mp4" 
    output_folder_name = "instant_power_tracking"

    # --- 3. PATHS & DATA LOADING ---
    config_path = project_root / 'config'
    params_file = config_path / 'params.yaml'
    grid_file = config_path / 'grid_coords_dipole.yaml' 
    if not grid_file.exists(): grid_file = config_path / 'grid_coords.yaml'
    
    video_path = project_root / 'videos' / video_filename
    results_dir = project_root / 'test results' / output_folder_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading params from: {params_file}")
    with open(params_file, 'r') as f: params = yaml.safe_load(f)
    with open(grid_file, 'r') as f: coords_data = yaml.safe_load(f)

    # --- 4. COORDINATE MAPPING ---
    x_cortex = np.array(coords_data['x'])
    y_cortex = np.array(coords_data['y'])
    n_total = len(x_cortex)
    
    # Visualization Coordinates (High Res)
    plot_res = (800, 800) 
    px_coords, py_coords = map_coords_to_pixels(x_cortex, y_cortex, img_size=plot_res, padding=60)
    
    # Simulator Filtering
    rng = np.random.default_rng(params['run']['seed'])
    x_c, y_c = cortex_models.add_noise(x_cortex, y_cortex, params['cortex_model']['noise_scale'], rng)
    
    # Dropout Mask
    dropout_rate = params['cortex_model']['dropout_rate']
    keep_indices = rng.choice(np.arange(n_total), int(n_total * (1 - dropout_rate)), replace=False)
    dropout_mask = np.zeros(n_total, dtype=bool)
    dropout_mask[keep_indices] = True
    
    # Visual Field Transformation
    cortex_to_visual = cortex_models.get_mapping_from_cortex_to_visual_field(params['cortex_model'])
    z_visual = cortex_to_visual(cartesian_to_complex(x_c, y_c))
    x_vis, y_vis = z_visual.real, z_visual.imag
    
    # FOV Mask
    fov = params['run']['view_angle'] / 2
    x_org, y_org = params['run']['origin']
    fov_mask = (x_vis >= x_org - fov) & (x_vis < x_org + fov) & \
               (y_vis >= y_org - fov) & (y_vis < y_org + fov)
    
    valid_mask = dropout_mask & fov_mask
    n_active = np.sum(valid_mask)
    
    # Initialize Simulator
    sim = GaussianSimulator(params, Map(x_vis[valid_mask], y_vis[valid_mask]))
    
    # --- 5. SIMULATION LOOP ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sim_fps = fps if fps > 0 else 30.0
    dt = 1.0 / sim_fps
    
    total_power_hist = []
    full_grid_history = [] # Stores raw data for .npy
    frame_indices = []
    
    frame_nr = 0
    target_res = tuple(params['run']['resolution'])
    global_max_power = 0.0

    print("Running Simulation...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break 
            frame_nr += 1
            
            # Process Frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = get_center_crop(gray)
            crop_resized = cv2.resize(crop, target_res, interpolation=cv2.INTER_AREA)
            img_norm = crop_resized.astype(np.float32) / 255.0
            
            # Update Physics
            stim_amp = sim.sample_stimulus(img_norm, rescale=True)
            sim.update(stim_amp, dt=dt)
            
            # Extract Power
            state = sim.get_state()
            p_tensor = state['instant_power']
            active_watts = p_tensor.detach().cpu().numpy().flatten()
            
            total_w = np.sum(active_watts)
            
            # Map back to full grid (N=2500)
            grid_watts = np.zeros(n_total)
            L = min(len(active_watts), np.sum(valid_mask))
            idx = np.where(valid_mask)[0][:L]
            grid_watts[idx] = active_watts[:L]
            
            # Update Max for Plotting
            if np.max(grid_watts) > global_max_power:
                global_max_power = np.max(grid_watts)
            
            # Store Data
            total_power_hist.append(total_w)
            full_grid_history.append(grid_watts)
            frame_indices.append(frame_nr)
            
            if frame_nr % 50 == 0:
                print(f"Frame {frame_nr}: Total {total_w*1000:.2f} mW")

    finally:
        cap.release()

    # --- 6. OUTPUT GENERATION ---
    print("\nGenerating Requested Outputs...")
    time_axis = np.array(frame_indices) / sim_fps
    
    # A. SAVE .NPY FILE
    npy_path = results_dir / 'power_per_electrode.npy'
    np.save(npy_path, np.array(full_grid_history) * 1000) # Save in mW
    print(f"[1/3] Saved Raw Data: {npy_path}")

    # B. SAVE TOTAL POWER PLOT (ONLY)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Total Instant Power (mW)', color=color)
    ax1.plot(time_axis, np.array(total_power_hist) * 1000, color=color, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Total Power Evolution: {video_filename}")

    stats_path = results_dir / 'total_power_evolution.png'
    plt.tight_layout()
    plt.savefig(stats_path, dpi=150)
    plt.close()
    print(f"[2/3] Saved Total Power Plot: {stats_path}")

    # C. GENERATE VIDEO (power_time_analysis.avi)
    print("[3/3] Rendering power_time_analysis.avi...")
    
    playback_cap = cv2.VideoCapture(str(video_path))
    combined_res = (1600, 800)
    video_out_path = results_dir / 'power_time_analysis.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_vid = cv2.VideoWriter(str(video_out_path), fourcc, sim_fps, combined_res, isColor=True)
    
    # Prep LUT
    lut_range = np.linspace(0, 255, 256).astype(np.uint8)
    lut = cv2.applyColorMap(np.expand_dims(lut_range, 1), cv2.COLORMAP_INFERNO).squeeze()
    
    # Prep Base Scatter (Inactive Electrodes)
    base_scatter = np.zeros((plot_res[1], plot_res[0], 3), dtype=np.uint8)
    for i in range(n_total):
        cv2.circle(base_scatter, (px_coords[i], py_coords[i]), 2, (30, 30, 30), -1)

    # Render
    global_max_mw = global_max_power * 1000
    if global_max_mw == 0: global_max_mw = 1.0

    for i in range(len(full_grid_history)):
        ret, raw_frame = playback_cap.read()
        if not ret: break
        
        # Left Side: Input
        left_img = get_center_crop(raw_frame, target_size=plot_res)
        
        # Right Side: Power Scatter
        right_img = base_scatter.copy()
        frame_watts = full_grid_history[i] * 1000 # Convert to mW
        
        # Normalize only valid active electrodes
        norm_p = (frame_watts / global_max_mw * 255).astype(int).clip(0, 255)
        
        active_idx = np.where(valid_mask)[0]
        for idx in active_idx:
            val = norm_p[idx]
            color = lut[val].tolist()
            # Size modulation based on intensity
            rad = 3 if val < 50 else (4 if val < 150 else 6)
            cv2.circle(right_img, (px_coords[idx], py_coords[idx]), rad, color, -1)
            
        # Combine
        combined = np.hstack([left_img, right_img])
        
        # Overlays
        cv2.putText(combined, "Input (Processed)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        stats = f"T={time_axis[i]:.2f}s | Total: {total_power_hist[i]*1000:.0f}mW"
        cv2.putText(combined, stats, (820, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        out_vid.write(combined)

    playback_cap.release()
    out_vid.release()
    print("✓ Done.")

if __name__ == "__main__":
    run_video_tracking()
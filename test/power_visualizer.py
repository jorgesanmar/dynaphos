import sys
import os
import cv2
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# --- SETUP PATHS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dynaphos import utils, cortex_models
from dynaphos.utils import Map

class PowerVisualizer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        # 1. Load Metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.meta = json.load(f)
            
        # 2. Load Raw Data
        print("Loading data files...")
        self.amplitudes = np.load(self.data_dir / "amplitudes.npy") # Shape: (Frames, Electrodes)
        self.impedances = np.load(self.data_dir / "impedances.npy") # Shape: (Electrodes,)
        
        # 3. Load Params Snapshot (for consistent physics)
        with open(self.data_dir / "params_snapshot.yaml", 'r') as f:
            self.params = yaml.safe_load(f)

        # 4. Setup Physics Constants
        self.fps = self.meta['video_fps']
        self.dt = 1.0 / self.fps
        
        # Calculate Duty Cycle (Pulse Train active time ratio)
        # Duty Cycle = PulseWidth * Frequency
        # (Assuming biphasic, we might multiply by 2, but power is I^2*R during the pulse)
        pw = self.meta['simulation_params']['pulse_width']
        freq = self.meta['simulation_params']['frequency']
        # Relative stim duration is usually 1.0 (continuous train during frame) or less
        rel_dur = self.params['default_stim'].get('relative_stim_duration', 1.0)
        
        # Fraction of time current is actually flowing
        self.duty_cycle = pw * freq * rel_dur
        
        # 5. Initialize Coordinates (for plotting)
        self._init_coordinates()

    def _init_coordinates(self):
        """Re-instantiates the coordinate map for visualization."""
        grid_file = project_root / 'config' / 'grid_coords_dipole_valid.yaml'
        if not grid_file.exists():
            grid_file = project_root / 'config' / 'grid_coords_dipole.yaml'
            
        x, y = utils.load_coordinates_from_yaml(str(grid_file))
        coordinates_cortex = Map(x=x, y=y)
        
        # Get Visual Field Coordinates
        try:
            phosphene_coords = cortex_models.get_visual_field_coordinates_from_cortex_full(
                self.params['cortex_model'], coordinates_cortex
            )
            self.x_coords = phosphene_coords._x
            self.y_coords = phosphene_coords._y
        except AttributeError:
            phosphene_coords = cortex_models.get_visual_field_coordinates_from_cortex(
                self.params['cortex_model'], coordinates_cortex
            )
            # Try getting x/y, fallback to _x/_y
            if hasattr(phosphene_coords, 'x'):
                self.x_coords, self.y_coords = phosphene_coords.x, phosphene_coords.y
            else:
                self.x_coords, self.y_coords = phosphene_coords._x, phosphene_coords._y

    def map_coords_to_pixels(self, img_size=(600, 600), padding=40):
        """Maps visual field coordinates to pixel coordinates for the output video."""
        min_x, max_x = np.min(self.x_coords), np.max(self.x_coords)
        min_y, max_y = np.min(self.y_coords), np.max(self.y_coords)
        
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        draw_w = img_size[0] - 2 * padding
        draw_h = img_size[1] - 2 * padding
        
        # Normalize and scale
        px = (padding + (self.x_coords - min_x) / range_x * draw_w).astype(int)
        py = (padding + (self.y_coords - min_y) / range_y * draw_h).astype(int)
        
        return px, py

    def calculate_physics(self, thermal_tau=2.0):
        """
        Calculates Power and Thermal Load offline.
        tau: Thermal time constant in seconds.
        """
        print(f"Calculating Physics (Tau={thermal_tau}s)...")
        
        # --- 1. INSTANT POWER (Peak during pulse) ---
        # P_peak = I^2 * R
        # amplitudes array is in Amperes
        self.power_inst_peak = (self.amplitudes ** 2) * self.impedances
        
        # --- 2. AVERAGE POWER (Per Frame) ---
        # P_avg = P_peak * Duty_Cycle
        self.power_avg = self.power_inst_peak * self.duty_cycle
        
        # --- 3. THERMAL LOAD (Leaky Integrator) ---
        # T[t] = T[t-1] * decay + Power_Input * dt
        # We use Average Power as input because heat accumulates over the whole frame duration
        
        num_frames, num_elecs = self.power_avg.shape
        self.thermal_load = np.zeros_like(self.power_avg)
        
        decay_factor = np.exp(-self.dt / thermal_tau)
        input_scale = self.dt  # Simple scaling: Temp rise proportional to Energy (Power * dt)
        
        # Iterative calculation (cannot easily vectorize the recurrence relation)
        current_temp = np.zeros(num_elecs)
        
        for t in range(num_frames):
            # Input power for this frame
            power_in = self.power_avg[t]
            
            # Update Temp
            current_temp = (current_temp * decay_factor) + (power_in * input_scale)
            
            # Store
            self.thermal_load[t] = current_temp
            
        print("Physics calculation complete.")

    def generate_plots(self):
        """Generates static analysis plots."""
        print("Generating plots...")
        time_axis = np.arange(len(self.amplitudes)) / self.fps
        
        # 1. Total Power Trace
        total_power = np.sum(self.power_avg, axis=1) * 1000 # Convert to mW
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, total_power, color='#d62728', linewidth=1.5)
        plt.title(f"Total Power Consumption (Average over Frame)\nSource: {self.meta['video_source']}")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Power (mW)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.data_dir / "plot_total_power.png", dpi=150)
        plt.close()

        # 2. Thermal Load Trace (Max Temperature Proxy)
        # Tracks the "hottest" electrode at any given time
        max_temp = np.max(self.thermal_load, axis=1)
        mean_temp = np.mean(self.thermal_load, axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, max_temp, label="Hottest Electrode", color='#ff7f0e')
        plt.plot(time_axis, mean_temp, label="Array Mean", color='#1f77b4', linestyle='--')
        plt.title(f"Thermal Load Simulation (Tau={self.params['temporal_dynamics'].get('thermal_time_constant', 2.0)}s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Thermal Load (Arbitrary Units)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.data_dir / "plot_thermal_load.png", dpi=150)
        plt.close()
        
    def render_video(self, output_filename="visual_analysis.avi"):
        """Renders the side-by-side visualization video."""
        video_path = project_root / 'videos' / self.meta['video_source']
        if not video_path.exists():
            print(f"⚠️ Original video not found at {video_path}. Skipping video generation.")
            return

        print(f"Rendering Video to {output_filename}...")
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Canvas Setup
        plot_res = (600, 600) # Size of the data plots
        total_w = plot_res[0] * 3 # Input | Power | Heat
        total_h = plot_res[1]
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_path = self.data_dir / output_filename
        out = cv2.VideoWriter(str(out_path), fourcc, self.fps, (total_w, total_h))
        
        # Pre-calculate coordinates
        px, py = self.map_coords_to_pixels(plot_res, padding=50)
        
        # Color Maps
        cmap_power = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256,1), cv2.COLORMAP_VIRIDIS).squeeze()
        cmap_heat = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256,1), cv2.COLORMAP_INFERNO).squeeze()
        
        # Normalization factors (Global max for stability)
        max_p = np.percentile(self.power_avg, 99.9) or 1.0
        max_t = np.max(self.thermal_load) or 1.0

        for t in tqdm(range(len(self.amplitudes)), desc="Rendering"):
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Left Panel: Input Video
            # Crop center and resize
            h, w = frame.shape[:2]
            s = min(h, w)
            start_y, start_x = (h-s)//2, (w-s)//2
            crop = frame[start_y:start_y+s, start_x:start_x+s]
            frame_resized = cv2.resize(crop, plot_res)
            
            # 2. Middle Panel: Instant Power (Scatter)
            # Shows WHICH electrodes are firing NOW
            panel_power = np.zeros((plot_res[1], plot_res[0], 3), dtype=np.uint8)
            power_frame = self.power_avg[t]
            
            # Vectorized drawing is hard in OpenCV, using loop
            # Normalize: 0 to 255
            indices = np.where(power_frame > 0)[0]
            if len(indices) > 0:
                vals = (power_frame[indices] / max_p * 255).clip(0, 255).astype(int)
                for i, val in zip(indices, vals):
                    color = cmap_power[val].tolist()
                    cv2.circle(panel_power, (px[i], py[i]), 4, color, -1)
            
            # 3. Right Panel: Thermal Load (Heatmap)
            # Shows accumulated heat (Smooth evolution)
            panel_heat = np.zeros((plot_res[1], plot_res[0], 3), dtype=np.uint8)
            temp_frame = self.thermal_load[t]
            
            indices_t = np.where(temp_frame > (max_t * 0.05))[0] # Threshold low values
            if len(indices_t) > 0:
                vals_t = (temp_frame[indices_t] / max_t * 255).clip(0, 255).astype(int)
                for i, val in zip(indices_t, vals_t):
                    color = cmap_heat[val].tolist()
                    # Draw larger circles for heat to simulate diffusion visually
                    cv2.circle(panel_heat, (px[i], py[i]), 8, color, -1)
            
            # Blur the heat panel to simulate diffusion
            panel_heat = cv2.GaussianBlur(panel_heat, (15, 15), 0)

            # Combine
            combined = np.hstack([frame_resized, panel_power, panel_heat])
            
            # Overlay Text
            cv2.putText(combined, "Input", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(combined, "Inst. Power Density", (620, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(combined, "Thermal Load (Accumulated)", (1220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            out.write(combined)
            
        cap.release()
        out.release()
        print(f"Done! Video saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize power tracking results.")
    parser.add_argument('--data-dir', type=str, required=True, 
                        help="Path to the data directory (e.g. data_logs/vid_name/config_name)")
    parser.add_argument('--tau', type=float, default=2.0, 
                        help="Thermal Time Constant (seconds) for leaky integrator")
    
    args = parser.parse_args()
    
    viz = PowerVisualizer(args.data_dir)
    
    # Run pipeline
    viz.calculate_physics(thermal_tau=args.tau)
    viz.generate_plots()
    viz.render_video()

if __name__ == "__main__":
    main()
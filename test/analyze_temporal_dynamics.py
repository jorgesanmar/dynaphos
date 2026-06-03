"""
Script to analyze temporal dynamics (fading/habituation) across multiple frequencies.
Saves videos, evolution plots per frequency, and a summary comparison across frequencies.
"""

import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import shutil
from pathlib import Path
from typing import Dict, List

import dynaphos.utils as utils
import dynaphos.cortex_models as cortex_models
from dynaphos.cortex_models import Map
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalDynamicsAnalyzer:
    def __init__(self, params_path: str, output_dir: str):
        self.params = utils.load_params(params_path)
        self.output_base = Path(output_dir)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Patterns to test
        self.patterns = ['no_raster', 'horizontal', 'vertical', 'checkerboard', 'random']
        
        # Load grid coordinates once
        base_dir = os.path.dirname(params_path) if os.path.dirname(params_path) else '.'
        coord_path = os.path.join(base_dir, '../config/grid_coords_dipole.yaml')
        
        if not os.path.exists(coord_path):
             coord_path = 'grid_coords_dipole.yaml'
        if not os.path.exists(coord_path):
             coord_path = '../config/grid_coords_dipole.yaml'

        x, y = utils.load_coordinates_from_yaml(coord_path)
        self.coords_cortex = Map(x=x, y=y)
        self.coords_visual = cortex_models.get_visual_field_coordinates_from_cortex_full(
            self.params['cortex_model'], self.coords_cortex
        )

    def run_multi_freq_analysis(self, frequencies: List[float], duration_s: float = 10.0):
        """Run analysis for multiple frequencies."""
        
        # Store retention rates for the final summary plot: {pattern: {freq: retention}}
        summary_retention = {p: [] for p in self.patterns}
        summary_freqs = sorted(frequencies)
        
        for freq in summary_freqs:
            logger.info(f"\n{'='*40}")
            logger.info(f"STARTING ANALYSIS FOR {freq} Hz")
            logger.info(f"{'='*40}")
            
            # Create subfolder for this frequency
            freq_dir = self.output_base / f"{freq}Hz"
            freq_dir.mkdir(exist_ok=True)
            
            # --- AUTO-ADJUST FPS ---
            # To simulate N groups switching at F Hz, we need fps >= N * F
            num_groups = self.params.get('raster', {}).get('num_groups', 5)
            required_fps = freq * num_groups
            
            # Add a safety buffer (e.g. 10%) or ensure minimum
            target_fps = max(required_fps, self.params['run']['fps'])
            # Round to nearest integer
            target_fps = int(np.ceil(target_fps))
            
            logger.info(f"  -> Auto-adjusting Simulation FPS to {target_fps} (Required for {freq}Hz * {num_groups} groups)")
            self.params['run']['fps'] = target_fps
            
            # Run simulation for this frequency
            results = self.run_simulation_batch(freq, target_fps, duration_s, freq_dir)
            
            # Calculate retention for summary
            avg_window = int(target_fps * 0.5)
            if avg_window < 1: avg_window = 1
            
            for pattern in self.patterns:
                data = results[pattern]
                start = data[:avg_window].mean()
                end = data[-avg_window:].mean()
                ret = (end/start * 100) if start > 0 else 0
                summary_retention[pattern].append(ret)
                
        # Generate the grand summary plot
        self.generate_freq_summary_plot(summary_freqs, summary_retention)

    def run_simulation_batch(self, freq: float, fps: int, duration_s: float, output_dir: Path):
        """Run simulation for a single frequency setting."""
        results = {}
        total_frames = int(duration_s * fps)
        res_x, res_y = self.params['run']['resolution']
        
        # Use a smaller image if FPS is huge to save disk space/processing time? 
        # No, keep resolution but user be warned videos will be large.
        white_frame = np.full((res_y, res_x), 255, dtype=np.uint8)
        
        for pattern in self.patterns:
            logger.info(f"  > Simulating {pattern}...")
            
            raster_enabled = (pattern != 'no_raster')
            r_conf = self.params.get('raster', {})
            num_groups = r_conf.get('num_groups', 5)
            
            simulator = PhospheneSimulator(
                self.params,
                self.coords_visual,
                raster_enabled=raster_enabled,
                raster_pattern=pattern if raster_enabled else 'checkerboard',
                raster_num_groups=num_groups,
                raster_rate_hz=freq # Override with current test freq
            )
            
            # Video Setup
            video_path = output_dir / f"{pattern}_sim.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (res_x * 2, res_y), False)
            
            history = []
            
            try:
                for _ in range(total_frames):
                    stim = simulator.sample_stimulus(white_frame, rescale=True)
                    output_tensor = simulator(stim)
                    
                    mean_b = output_tensor.mean().item()
                    history.append(mean_b)
                    
                    # Save Video Frame
                    phs_np = (output_tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                    frame_out = np.concatenate([white_frame, phs_np], axis=1)
                    out.write(frame_out)
                    
            finally:
                out.release()
                del simulator
                torch.cuda.empty_cache()
                
            results[pattern] = np.array(history)
            
        # Generate plots specific to this frequency
        self.generate_single_freq_plots(results, duration_s, fps, output_dir)
        return results

    def generate_single_freq_plots(self, results, duration, fps, output_dir):
        """Generate the standard temporal evolution plots for one frequency."""
        time_axis = np.linspace(0, duration, len(next(iter(results.values()))))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        window_size = int(fps * 0.5)
        
        for (pattern, data), color in zip(results.items(), colors):
            if window_size > 1:
                smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                t_smooth = time_axis[window_size-1:]
            else:
                smoothed = data
                t_smooth = time_axis
            
            # Absolute
            ax1.plot(t_smooth, smoothed, label=pattern.capitalize(), color=color, linewidth=2)
            
            # Proportional
            if len(smoothed) > 0 and smoothed[0] > 0:
                norm = (smoothed / smoothed[0]) * 100
                ax2.plot(t_smooth, norm, label=pattern.capitalize(), color=color, linewidth=2)
                
        ax1.set_title("Absolute Brightness", fontsize=12, fontweight='bold')
        ax2.set_title("Proportional (% of Initial)", fontsize=12, fontweight='bold')
        for ax in [ax1, ax2]:
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.suptitle(f"Temporal Dynamics @ {output_dir.name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "evolution_plot.png", dpi=150)
        plt.close()

    def generate_freq_summary_plot(self, freqs: List[float], retention_data: Dict[str, List[float]]):
        """Generate a plot comparing retention across frequencies."""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(retention_data)))
        
        for (pattern, rates), color in zip(retention_data.items(), colors):
            ax.plot(freqs, rates, marker='o', linewidth=2, label=pattern.capitalize(), color=color)
            
        ax.set_title("Brightness Retention vs. Raster Frequency", fontsize=14, fontweight='bold')
        ax.set_xlabel("Raster Frequency (Hz)")
        ax.set_ylabel("Brightness Retention (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save in the base directory
        plt.tight_layout()
        plt.savefig(self.output_base / "FREQUENCY_COMPARISON.png", dpi=150)
        plt.close()
        logger.info(f"Saved summary comparison to {self.output_base / 'FREQUENCY_COMPARISON.png'}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Phosphene Temporal Dynamics')
    parser.add_argument('--params', type=str, default='../config/params.yaml', help='Path to params')
    parser.add_argument('--output', type=str, default='./temporal_analysis', help='Output dir')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration in seconds')
    
    # New arg for frequencies
    parser.add_argument('--frequencies', type=float, nargs='+', default=[5.0, 10.0, 20.0, 40.0],
                        help='List of raster frequencies to test (e.g. 5 10 20)')
    
    args = parser.parse_args()
    
    analyzer = TemporalDynamicsAnalyzer(args.params, args.output)
    analyzer.run_multi_freq_analysis(args.frequencies, args.duration)
    
    logger.info("Analysis complete.")

if __name__ == '__main__':
    main()
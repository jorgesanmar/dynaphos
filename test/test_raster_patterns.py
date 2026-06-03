"""
Comprehensive test script for raster pattern strategies in phosphene vision simulation.

This script tests 4 raster patterns plus a no-raster baseline:
- Horizontal
- Vertical  
- Checkerboard
- Random
- No Raster (baseline)

For each pattern, it:
1. Processes a video file
2. Tracks cumulative charge per electrode
3. Saves output videos with phosphene simulation
4. Generates analytical frames showing which electrodes are active
5. Produces summary statistics and visualizations
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch

import dynaphos
from dynaphos import utils, cortex_models
from dynaphos.cortex_models import Map
from dynaphos.image_processing import canny_processor, sobel_processor, scale_image
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from dynaphos.utils import get_data_kwargs, to_numpy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RasterPatternTester:
    """Test harness for evaluating raster pattern strategies."""
    
    def __init__(self, params_path: str, video_path: str, output_dir: str,
                 edge_detection: str = 'none'):
        """
        Initialize the tester.
        
        Parameters
        ----------
        params_path : str
            Path to params.yaml configuration file
        video_path : str
            Path to input video file
        output_dir : str
            Directory for output files
        edge_detection : str
            Edge detection method: 'none', 'canny', 'sobel'
        """
        self.params_path = params_path
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.edge_detection = edge_detection.lower()
        
        # Validate edge detection option
        valid_options = ['none', 'canny', 'sobel']
        if self.edge_detection not in valid_options:
            raise ValueError(f"edge_detection must be one of {valid_options}")
        
        # Load parameters
        self.params = utils.load_params(params_path)
        self.framerate = self.params['run']['fps']


        # Test configurations
        self.raster_patterns = ['horizontal', 'vertical', 'checkerboard', 'random', 'no_raster']
        self.results = {}
        
        logger.info(f"Initialized RasterPatternTester")
        logger.info(f"Video: {video_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"FPS: {self.framerate}")
        logger.info(f"Edge detection: {self.edge_detection}")
    
    def run_all_tests(self, max_frames: int = None):
        """
        Run tests for all raster patterns.
        
        Parameters
        ----------
        max_frames : int, optional
            Maximum number of frames to process. If None, process entire video.
        """
        logger.info("=" * 80)
        logger.info("STARTING RASTER PATTERN TESTS")
        logger.info("=" * 80)
        
        for pattern in self.raster_patterns:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing pattern: {pattern.upper()}")
            logger.info(f"{'='*80}")
            
            try:
                self.results[pattern] = self.test_pattern(pattern, max_frames)
                logger.info(f"✓ {pattern} test completed successfully")
            except Exception as e:
                logger.error(f"✗ {pattern} test failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate comparison report
        self.generate_comparison_report()
        
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS COMPLETED")
        logger.info("="*80)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a video frame with optional edge detection.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame (grayscale)
            
        Returns
        -------
        processed : np.ndarray
            Preprocessed frame ready for stimulation
        """
        # Resize and blur
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        
        # Apply edge detection if requested
        if self.edge_detection == 'canny':
            # Canny edge detection
            # Thresholds calibrated for good edge visibility
            processed = canny_processor(frame, threshold_low=20, threshold_high=75)
        elif self.edge_detection == 'sobel':
            # Sobel edge detection
            processed = sobel_processor(frame)
            # Normalize to 0-255 range
            processed = scale_image(processed, f=255.0, use_max=True)
            processed = processed.astype(np.uint8)
        else:
            # No edge detection - use blurred frame directly
            processed = frame
        
        return processed
    
    def test_pattern(self, pattern: str, max_frames: int = None) -> Dict:
        """
        Test a single raster pattern.
        """
        import gc
        import torch

        # --- Initialization ---
        raster_enabled = (pattern != 'no_raster')
        raster_pattern = pattern if raster_enabled else 'checkerboard'
        
        raster_config = self.params.get('raster', {})
        num_groups = raster_config.get('num_groups', 5)
        rate_hz = raster_config.get('rate_hz', 4.5)

        x,y = utils.load_coordinates_from_yaml(os.path.join('../config/grid_coords_dipole.yaml'))
        coordinates_cortex = Map(x=x,y=y)
        phosphene_coords = cortex_models.get_visual_field_coordinates_from_cortex_full(self.params['cortex_model'], coordinates_cortex)
        
        simulator = PhospheneSimulator(
            self.params,
            phosphene_coords,
            raster_enabled=raster_enabled,
            raster_pattern=raster_pattern,
            raster_num_groups=num_groups,
            raster_rate_hz=rate_hz
        )
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is None: max_frames = total_frames
        else: max_frames = min(max_frames, total_frames)
        
        output_video_path = self.output_dir / f"{pattern}_phosphenes.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_video_path), fourcc, self.framerate, (512, 256), False)

        # --- Processing Loop ---
        charge_history = []
        sample_frames = []
        sample_frame_numbers = []
        active_electrode_frames = []
        
        captured_groups = set()
        
        frame_nr = 0
        try:
            while frame_nr < max_frames:
                ret, frame = cap.read()
                if not ret: break
                frame_nr += 1
                
                # Preprocessing
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.shape[0] != frame.shape[1]:
                    s = min(frame.shape)
                    frame = frame[frame.shape[0]//2-s//2:frame.shape[0]//2+s//2, 
                                frame.shape[1]//2-s//2:frame.shape[1]//2+s//2]
                
                processed_img = self.preprocess_frame(frame)
                stim_pattern = simulator.sample_stimulus(processed_img, rescale=True)
                
                # Generate phosphenes
                phs = simulator(stim_pattern).clamp(0, 1)
                phs_np = to_numpy(phs) * 255
                
                # Record charge status
                charge_status = simulator.get_charge_status()
                
                # --- FIXED ACTIVE COUNT LOGIC ---
                # We check effective_charge_per_second to see who ACTUALLY fired.
                # This accounts for image content + raster mask + leak current.
                eff_charge = simulator.effective_charge_per_second
                if eff_charge is not None:
                    # Count electrodes delivering > 0 charge
                    active_count = int((eff_charge > 0).sum().item())
                else:
                    active_count = 0

                charge_history.append({
                    'frame': frame_nr,
                    'total_charge_uC': charge_status['total_charge_uC'],
                    'mean_charge_uC': charge_status['cumulative_charge_uC'].mean().item(),
                    'active_count': active_count
                })
                
                # Save sample frames
                if raster_enabled:
                    current_group = simulator.current_raster_group
                    if current_group not in captured_groups:
                        sample_frames.append({
                            'input': processed_img.copy(),
                            'output': phs_np.copy(),
                            'raster_group': current_group
                        })
                        sample_frame_numbers.append(frame_nr)
                        captured_groups.add(current_group)
                        
                        active_viz = self.visualize_active_electrodes(
                            simulator, stim_pattern, phosphene_coords 
                        )
                        active_electrode_frames.append(active_viz)
                else:
                    if frame_nr <= 5:
                        sample_frames.append({
                            'input': processed_img.copy(),
                            'output': phs_np.copy(),
                            'raster_group': None
                        })
                        sample_frame_numbers.append(frame_nr)
                
                # Write video
                cat = np.concatenate([processed_img, phs_np], axis=1).astype('uint8')
                out.write(cat)
                
                if frame_nr % 30 == 0:
                    logger.info(f"Frame {frame_nr}/{max_frames}: Total Charge {charge_status['total_charge_uC']:.1f} µC")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        # Final stats
        final_charge_status = simulator.get_charge_status()
        final_cumulative = final_charge_status['cumulative_charge_uC'].cpu().numpy()
        
        results = {
            'pattern': pattern,
            'frames_processed': frame_nr,
            'final_cumulative_charge': final_cumulative,
            'electrode_shape': simulator.electrode_array_shape,
            'mean_cumulative_charge': final_cumulative.mean(),
            'max_cumulative_charge': final_charge_status['max_charge_uC'],
            'min_cumulative_charge': final_charge_status['min_charge_uC'],
            'max_electrode_idx': final_charge_status['max_electrode_idx'],
            'total_cumulative_charge': final_charge_status['total_charge_uC'],
            'charge_limit': final_charge_status['limit_uC'],
            'charge_history': charge_history,
            'sample_frames': sample_frames,
            'sample_frame_numbers': sample_frame_numbers,
            'active_electrode_frames': active_electrode_frames,
            'output_video_path': str(output_video_path),
            'raster_enabled': raster_enabled,
            'num_groups': num_groups if raster_enabled else None,
            'rate_hz': rate_hz if raster_enabled else None
        }
        
        self.generate_pattern_visualizations(results)
        
        # --- CLEANUP ---
        del simulator
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    def visualize_active_electrodes(
        self,
        simulator: PhospheneSimulator,
        stim_pattern: torch.Tensor,
        phosphene_coords: cortex_models.Map
    ) -> np.ndarray:
        """
        Create scatter plot visualization showing active electrodes in the Visual Field.
        """
        if not simulator.raster_enabled:
            # Simple placeholder if raster is off
            return np.full((400, 400, 3), 200, dtype=np.uint8)
        
        # 1. Get raster mask (1D array of 0s and 1s)
        raster_mask = simulator.get_current_raster_mask()
        mask_np = utils.to_numpy(raster_mask.squeeze())
        
        # 2. Get Visual Field coordinates
        # These are already filtered by the simulator to match mask_np dimensions
        x, y = phosphene_coords.cartesian
        
        # 3. Setup Canvas (Square image)
        canvas_size = 400
        vis = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # 4. Normalize coordinates to image pixels (0 to canvas_size)
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1
        
        x_norm = (x - x.min()) / x_range
        y_norm = (y - y.min()) / y_range
        
        # Convert to pixel indices with padding
        # Note: We map y_norm (0..1) directly to pixel index (0..H).
        # In image coords, 0 is Top. So y.min() maps to Top. 
        # This matches the simulator's raster logic where 'low index' is top-left.
        padding = 20
        scale = canvas_size - (2 * padding)
        px = (x_norm * scale).astype(int) + padding
        py = (y_norm * scale).astype(int) + padding
        
        # 5. Draw Electrodes
        # Draw ALL electrodes as faint gray dots (background)
        for i in range(len(px)):
            cv2.circle(vis, (px[i], py[i]), 2, (50, 50, 50), -1) 
            
        # Draw ACTIVE electrodes as bright green dots
        active_indices = np.where(mask_np > 0.5)[0]
        for i in active_indices:
            if i < len(px):
                cv2.circle(vis, (px[i], py[i]), 4, (0, 255, 0), -1)
        
        return vis
    
    def generate_pattern_visualizations(self, results: Dict):
        """Generate the specific requested visualizations."""
        pattern = results['pattern']
        logger.info(f"Generating visualizations for {pattern}...")
        
        fig_dir = self.output_dir / f"{pattern}_figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Total Cumulative Charge Evolution
        self._plot_total_charge_evolution(results, fig_dir)
        
        # 2. Charge Distribution & Heatmap
        self._plot_charge_distribution_and_heatmap(results, fig_dir)
        
        # 3. Mean Cumulative Charge Evolution
        self._plot_mean_charge_evolution(results, fig_dir)
        
        # 4. Active Electrodes Evolution
        self._plot_active_count_evolution(results, fig_dir)
        
        # 5. Sample Frames (Using the corrected version)
        self._plot_sample_frames(results, fig_dir)

    def generate_pattern_visualizations(self, results: Dict):
        """Generate the specific requested visualizations."""
        pattern = results['pattern']
        logger.info(f"Generating visualizations for {pattern}...")
        
        fig_dir = self.output_dir / f"{pattern}_figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Total Cumulative Charge Evolution
        self._plot_total_charge_evolution(results, fig_dir)
        
        # 2. Charge Distribution & Heatmap
        self._plot_charge_distribution_and_heatmap(results, fig_dir)
        
        # 3. Mean Cumulative Charge Evolution
        self._plot_mean_charge_evolution(results, fig_dir)
        
        # 4. Active Electrodes Evolution
        self._plot_active_count_evolution(results, fig_dir)
        
        # 5. Sample Frames
        self._plot_sample_frames(results, fig_dir)

    def _plot_total_charge_evolution(self, results: Dict, output_dir: Path):
        """Plot evolution of the sum of charge of all electrodes."""
        history = results['charge_history']
        frames = [h['frame'] for h in history]
        total = [h['total_charge_uC'] for h in history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frames, total, color='purple', linewidth=2)
        ax.set_title(f"Total Cumulative Charge Evolution - {results['pattern'].upper()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Total Charge (µC)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'total_charge_evolution.png', dpi=150)
        plt.close()

    def _plot_charge_distribution_and_heatmap(self, results: Dict, output_dir: Path):
        """Histogram (left) and Color Coded Matrix (right) of final charges."""
        charges = results['final_cumulative_charge']
        mean_val = results['mean_cumulative_charge']
        max_val = results['max_cumulative_charge']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Left: Histogram ---
        ax1.hist(charges, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add Mean and Max Lines (No Limit Line)
        ax1.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean ({mean_val:.2f} µC)')
        ax1.axvline(max_val, color='orange', linestyle='--', linewidth=2, label=f'Max ({max_val:.2f} µC)')
        
        ax1.set_title("Final Charge Distribution", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Cumulative Charge (µC)")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # --- Right: Heatmap (Matrix) ---
        target_shape = results['electrode_shape'] 
        
        if charges.size != (target_shape[0] * target_shape[1]):
            side = int(np.ceil(np.sqrt(charges.size)))
            target_shape = (side, side)
            padded = np.zeros(side * side)
            padded[:charges.size] = charges
            matrix = padded.reshape(target_shape)
        else:
            matrix = charges.reshape(target_shape)
            
        im = ax2.imshow(matrix, cmap='inferno', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Cumulative Charge (µC)')
        
        ax2.set_title("Electrode Charge Heatmap", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Electrode Column")
        ax2.set_ylabel("Electrode Row")
        
        plt.suptitle(f"Charge Analysis - {results['pattern'].upper()}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'charge_distribution_heatmap.png', dpi=150)
        plt.close()

    def _plot_mean_charge_evolution(self, results: Dict, output_dir: Path):
        """Plot evolution of mean cumulative charge."""
        history = results['charge_history']
        frames = [h['frame'] for h in history]
        means = [h['mean_charge_uC'] for h in history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frames, means, color='green', linewidth=2)
        ax.set_title(f"Mean Cumulative Charge Evolution - {results['pattern'].upper()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Mean Charge per Electrode (µC)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'mean_charge_evolution.png', dpi=150)
        plt.close()

    def _plot_active_count_evolution(self, results: Dict, output_dir: Path):
        """Plot evolution of active electrodes (averaged over 5-frame blocks)."""
        history = results['charge_history']
        
        # Extract data
        frames = np.array([h['frame'] for h in history])
        counts = np.array([h['active_count'] for h in history])
        
        # Define block size
        k = 5
        
        # Truncate data to be divisible by k
        n_blocks = len(frames) // k
        if n_blocks > 0:
            limit = n_blocks * k
            counts_trunc = counts[:limit]
            frames_trunc = frames[:limit]
            
            # Compute mean for each block
            # Reshape to (n_blocks, k) and average along the second axis (frames in block)
            counts_avg = counts_trunc.reshape(-1, k).mean(axis=1)
            
            # Use the time of the LAST frame in each block for the X-axis
            # (e.g., for block 1-5, plot at frame 5)
            frames_ds = frames_trunc[k-1::k]
            time_ds = frames_ds / self.framerate
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_ds, counts_avg, color='orange', linewidth=2)
            ax.fill_between(time_ds, counts_avg, color='orange', alpha=0.2)
            
            ax.set_title(f"Active Electrodes (Mean of 5-Frame Blocks) - {results['pattern'].upper()}", 
                         fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Avg Active Electrodes")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'active_count_evolution.png', dpi=150)
            plt.close()

    def _plot_sample_frames(self, results: Dict, output_dir: Path):
        """Plot sample frames with clean formatting."""
        sample_frames = results['sample_frames']
        active_frames = results.get('active_electrode_frames', [])
        frame_numbers = results['sample_frame_numbers']
        
        if not sample_frames:
            return
        
        # Sort by raster group number if available
        if sample_frames and sample_frames[0]['raster_group'] is not None:
            combined = list(zip(sample_frames, active_frames or [None]*len(sample_frames), frame_numbers))
            combined.sort(key=lambda x: x[0]['raster_group'])
            sample_frames, active_frames, frame_numbers = zip(*combined)
            active_frames = [a for a in active_frames if a is not None]
        
        n_samples = len(sample_frames)
        has_active = len(active_frames) == n_samples
        
        cols = 3 if has_active else 2
        
        # Adjust figure size
        fig = plt.figure(figsize=(cols * 4, n_samples * 3.5))
        gs = GridSpec(n_samples, cols, figure=fig, wspace=0.1, hspace=0.1)
        
        for i, (frame_data, frame_num) in enumerate(zip(sample_frames, frame_numbers)):
            raster_group = frame_data.get('raster_group', None)
            
            # --- Column 1: Input ---
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(frame_data['input'], cmap='gray', vmin=0, vmax=255)
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Row Title (Left of image)
            if raster_group is not None:
                ax1.set_ylabel(f"Group {raster_group}", fontsize=12, fontweight='bold')
            else:
                ax1.set_ylabel(f"Frame {frame_num}", fontsize=12)
                
            # Column Title (Top row only)
            if i == 0:
                ax1.set_title(f"Input ({self.edge_detection})", fontsize=14, fontweight='bold', pad=10)
            
            # --- Column 2: Phosphenes ---
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(frame_data['output'], cmap='gray', vmin=0, vmax=255)
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            if i == 0:
                ax2.set_title("Phosphenes", fontsize=14, fontweight='bold', pad=10)
            
            # --- Column 3: Activated Group ---
            if has_active:
                ax3 = fig.add_subplot(gs[i, 2])
                # No flipud here - ensures Group 0 (Top) appears at Top
                ax3.imshow(active_frames[i]) 
                ax3.set_xticks([])
                ax3.set_yticks([])
                
                if i == 0:
                    ax3.set_title("Activated Group", fontsize=14, fontweight='bold', pad=10)
        
        # Main Figure Title
        plt.suptitle(
            f"Sample Frames - {results['pattern'].upper()}",
            fontsize=16,
            fontweight='bold',
            y=0.95
        )
        
        # Save
        plt.savefig(output_dir / 'sample_frames.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self):
        """Generate comparison plots across all patterns."""
        logger.info("\nGenerating comparison report...")
        if not self.results: return
        
        patterns = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(patterns)))
        
        # Figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # --- Common Data Extraction ---
        plot_data = {}
        k = 5  # Block size
        
        for p in patterns:
            history = self.results[p]['charge_history']
            
            frames = np.array([x['frame'] for x in history])
            total = np.array([x['total_charge_uC'] for x in history])
            mean = np.array([x['mean_charge_uC'] for x in history])
            active = np.array([x['active_count'] for x in history])
            
            # Data Preparation
            # 1. For Cumulative/Mean Charge: Simple Downsampling (Every kth frame)
            indices = slice(0, len(frames), k)
            time_s = frames[indices] / self.framerate
            total_ds = total[indices]
            mean_ds = mean[indices]
            
            # 2. For Active Count: Block Averaging
            # Truncate to multiple of k
            n_blocks = len(frames) // k
            if n_blocks > 0:
                limit = n_blocks * k
                active_trunc = active[:limit]
                # Calculate mean of every 5 frames
                active_avg = active_trunc.reshape(-1, k).mean(axis=1)
                # Align time axis to the active_avg (using matching length)
                # We use the time points corresponding to the end of each block
                frames_trunc = frames[:limit]
                time_avg = frames_trunc[k-1::k] / self.framerate
            else:
                active_avg = active # Fallback if video is tiny
                time_avg = time_s

            plot_data[p] = {
                'time_ds': time_s,
                'total': total_ds,
                'mean': mean_ds,
                'time_avg': time_avg,
                'active_avg': active_avg
            }

        # 1. Top Left: Total Cumulative Charge Evolution
        ax = axes[0, 0]
        for p, c in zip(patterns, colors):
            d = plot_data[p]
            ax.plot(d['time_ds'], d['total'], label=p, color=c, linewidth=2)
        ax.set_title("Total Cumulative Charge Evolution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total Charge (µC)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Top Right: Mean Cumulative Charge Evolution
        ax = axes[0, 1]
        for p, c in zip(patterns, colors):
            d = plot_data[p]
            ax.plot(d['time_ds'], d['mean'], label=p, color=c, linewidth=2)
        ax.set_title("Mean Cumulative Charge Evolution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Charge (µC)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Bottom Left: Active Electrodes Count (Averaged)
        ax = axes[1, 0]
        for p, c in zip(patterns, colors):
            d = plot_data[p]
            ax.plot(d['time_avg'], d['active_avg'], label=p, color=c, linewidth=1.5)
        ax.set_title("Active Electrodes (Mean of 5-Frame Blocks)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Avg Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Bottom Right: Boxplot of Final Charge Distribution
        ax = axes[1, 1]
        charge_distributions = [self.results[p]['final_cumulative_charge'] for p in patterns]
        
        bp = ax.boxplot(
            charge_distributions,
            labels=patterns,
            patch_artist=True,
            showmeans=True
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        ax.set_title("Final Charge Distribution", fontsize=12, fontweight='bold')
        ax.set_ylabel("Cumulative Charge (µC)")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle("Raster Pattern Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_report.png', dpi=150)
        plt.close()
        
        self._generate_text_summary()
    
    def _generate_text_summary(self):
        """Generate text summary of results."""
        summary_path = self.output_dir / 'summary_report.txt'
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RASTER PATTERN COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for pattern in self.results:
                r = self.results[pattern]
                f.write(f"\n{pattern.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Frames processed: {r['frames_processed']}\n")
                f.write(f"Mean cumulative charge: {r['mean_cumulative_charge']:.2f} µC\n")
                f.write(f"Max cumulative charge: {r['max_cumulative_charge']:.2f} µC\n")
                f.write(f"Max electrode index: {r['max_electrode_idx']}\n")
                f.write(f"Min cumulative charge: {r['min_cumulative_charge']:.2f} µC\n")    
                f.write(f"Total cumulative charge: {r['total_cumulative_charge']:.2f} µC\n")
                f.write(f"Charge limit: {r['charge_limit']:.2f} µC\n")
                
                if r['raster_enabled']:
                    f.write(f"Number of groups: {r['num_groups']}\n")
                    f.write(f"Raster rate: {r['rate_hz']} Hz\n")
                
                f.write(f"Output video: {r['output_video_path']}\n")
            
            # Comparison
            f.write("\n" + "=" * 80 + "\n")
            f.write("COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            # Rank by mean charge
            ranked_mean = sorted(
                self.results.items(),
                key=lambda x: x[1]['mean_cumulative_charge']
            )
            f.write("Patterns ranked by mean cumulative charge (lowest to highest):\n")
            for i, (pattern, r) in enumerate(ranked_mean, 1):
                f.write(f"  {i}. {pattern.upper()}: {r['mean_cumulative_charge']:.2f} µC\n")
            
            # Rank by max charge
            ranked_max = sorted(
                self.results.items(),
                key=lambda x: x[1]['max_cumulative_charge']
            )
            f.write("\nPatterns ranked by max cumulative charge (lowest to highest):\n")
            for i, (pattern, r) in enumerate(ranked_max, 1):
                f.write(f"  {i}. {pattern.upper()}: {r['max_cumulative_charge']:.2f} µC\n")
            
            #Rank by total charge
            ranked_total = sorted(
                self.results.items(),
                key=lambda x: x[1]['total_cumulative_charge']
            )
            f.write("\nPatterns ranked by total cumulative charge (lowest to highest):\n")
            for i, (pattern, r) in enumerate(ranked_total, 1):
                f.write(f"  {i}. {pattern.upper()}: {r['total_cumulative_charge']:.2f} µC\n")
        
        logger.info(f"Summary report saved to: {summary_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test raster patterns for phosphene vision simulation'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='params.yaml',
        help='Path to params.yaml file'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./raster_test_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )
    parser.add_argument(
        '--edge-detection',
        type=str,
        default='none',
        choices=['none', 'canny', 'sobel'],
        help='Edge detection method: none (default), canny, or sobel'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = RasterPatternTester(
        params_path=args.params,
        video_path=args.video,
        output_dir=args.output,
        edge_detection=args.edge_detection
    )
    
    tester.run_all_tests(max_frames=args.max_frames)
    
    logger.info("\n✓ All tests completed successfully!")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Edge detection method: {args.edge_detection}")


if __name__ == '__main__':
    main()

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
from dynaphos.image_processing import canny_processor, sobel_processor
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
        frame = cv2.GaussianBlur(frame, (9, 9), 5)

        
        # Apply edge detection if requested
        if self.edge_detection == 'canny':
            # Canny edge detection
            # Thresholds calibrated for good edge visibility
            processed = canny_processor(frame, threshold_low=50, threshold_high=150)
        elif self.edge_detection == 'sobel':
            # Sobel edge detection
            processed = sobel_processor(frame)
            # Normalize to 0-255 range
            processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
            processed = processed.astype(np.uint8)
        else:
            # No edge detection - use blurred frame directly
            processed = frame
        
        return processed
    
    def test_pattern(self, pattern: str, max_frames: int = None) -> Dict:
        """
        Test a single raster pattern.
        
        Parameters
        ----------
        pattern : str
            Raster pattern name
        max_frames : int, optional
            Maximum frames to process
            
        Returns
        -------
        results : dict
            Test results including charge statistics and frame data
        """
        # Initialize simulator with appropriate raster settings
        raster_enabled = (pattern != 'no_raster')
        raster_pattern = pattern if raster_enabled else 'checkerboard'
        
        # Get raster settings from params or use defaults
        raster_config = self.params.get('raster', {})
        num_groups = raster_config.get('num_groups', 5)
        rate_hz = raster_config.get('rate_hz', 4.5)

       
       

        # Use probabilistic scatter for random and no_raster
        n_phosphenes = 1024
        phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(
            self.params, n_phosphenes
        )
        logger.info(f"  Total electrodes: {n_phosphenes}")
        
        simulator = PhospheneSimulator(
            self.params,
            phosphene_coords,
            raster_enabled=raster_enabled,
            raster_pattern=raster_pattern,
            raster_num_groups=num_groups,
            raster_rate_hz=rate_hz
        )
        
        logger.info(f"Simulator initialized:")
        logger.info(f"  - Raster enabled: {raster_enabled}")
        if raster_enabled:
            logger.info(f"  - Pattern: {raster_pattern}")
            logger.info(f"  - Groups: {num_groups}")
            logger.info(f"  - Rate: {rate_hz} Hz")
            raster_info = simulator.get_raster_info()
            logger.info(f"  - Raster info: {raster_info}")
        
        # Prepare video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is None:
            max_frames = total_frames
        else:
            max_frames = min(max_frames, total_frames)
        
        logger.info(f"Processing {max_frames} frames...")
        
        # Prepare output video
        output_video_path = self.output_dir / f"{pattern}_phosphenes.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            self.framerate,
            (512, 256),
            False
        )
        
        # Storage for analysis
        charge_history = []
        sample_frames = []
        sample_frame_numbers = []
        active_electrode_frames = []
        
        # Track which raster groups we've captured
        captured_groups = set()
        
        # Process frames
        frame_nr = 0
        while frame_nr < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_nr += 1
            
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get square crop if needed
            if frame.shape[0] != frame.shape[1]:
                shortest_side = min(frame.shape)
                h, w = frame.shape
                frame = frame[
                    h//2 - shortest_side//2 : h//2 + shortest_side//2,
                    w//2 - shortest_side//2 : w//2 + shortest_side//2
                ]
            
            # Apply preprocessing (resize, blur, optional edge detection)
            processed_img = self.preprocess_frame(frame)
            stim_pattern = simulator.sample_stimulus(processed_img, rescale=True)
            stim_pattern = stim_pattern*3.0
            # Generate phosphenes
            phs = simulator(stim_pattern).clamp(0, 1)
            phs_np = to_numpy(phs) * 255
            
            # Record charge status
            charge_status = simulator.get_charge_status()

            charge_history.append({
                'frame': frame_nr,
                'max_charge_uC': charge_status['max_charge_uC'],
                'min_charge_uC': charge_status['min_charge_uC'],      
                'total_charge_uC': charge_status['total_charge_uC'],  
                'max_electrode_idx': charge_status['max_electrode_idx'],
                'mean_charge_uC': charge_status['cumulative_charge_uC'].mean().item(),
                'cumulative_charge': charge_status['cumulative_charge_uC'].clone()
            })
            
            # Save sample frames (one per unique raster group)
            if raster_enabled:
                current_group = simulator.current_raster_group
                
                # Save frame if we haven't captured this group yet
                if current_group not in captured_groups:
                    sample_frames.append({
                        'input': processed_img.copy(),
                        'output': phs_np.copy(),
                        'raster_group': current_group
                    })
                    sample_frame_numbers.append(frame_nr)
                    captured_groups.add(current_group)
                    
                    # Create visualization of active electrodes
                    active_viz = self.visualize_active_electrodes(
                        simulator, stim_pattern
                    )
                    active_electrode_frames.append(active_viz)
                    
                    # Stop collecting once we have all groups
                    if len(captured_groups) >= num_groups:
                        logger.info(f"Captured all {num_groups} raster groups by frame {frame_nr}")
            else:
                # For no-raster, just save first 5 frames
                if frame_nr <= 5:
                    sample_frames.append({
                        'input': processed_img.copy(),
                        'output': phs_np.copy(),
                        'raster_group': None
                    })
                    sample_frame_numbers.append(frame_nr)
            
            # Write to output video
            cat = np.concatenate([processed_img, phs_np], axis=1).astype('uint8')
            out.write(cat)
            
            # Log progress
            if frame_nr % 30 == 0:
                logger.info(
                    f"Frame {frame_nr}/{max_frames}: "
                    f"Electrode {charge_status['max_electrode_idx']} = "
                    f"{charge_status['max_charge_uC']:.1f} µC / "
                    f"{charge_status['limit_uC']} µC"
                )
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final charge status
        final_charge_status = simulator.get_charge_status()
        final_cumulative = final_charge_status['cumulative_charge_uC'].cpu().numpy()
        
        logger.info(f"\nFinal charge statistics for {pattern}:")
        logger.info(f"  Max charge: {final_charge_status['max_charge_uC']:.2f} µC")
        logger.info(f"  Min charge: {final_charge_status['min_charge_uC']:.2f} µC")     
        logger.info(f"  Total charge: {final_charge_status['total_charge_uC']:.2f} µC") 
        logger.info(f"  Max electrode: {final_charge_status['max_electrode_idx']}")
        logger.info(f"  Mean charge: {final_cumulative.mean():.2f} µC")
        logger.info(f"  Charge limit: {final_charge_status['limit_uC']} µC")
        
        # Compile results
        results = {
            'pattern': pattern,
            'frames_processed': frame_nr,
            'final_cumulative_charge': final_cumulative,
            'mean_cumulative_charge': final_cumulative.mean(),
            'min_cumulative_charge': final_charge_status['min_charge_uC'],
            'max_cumulative_charge': final_charge_status['max_charge_uC'],
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
        
        # Generate visualizations for this pattern
        self.generate_pattern_visualizations(results)
        
        return results
    
    def visualize_active_electrodes(
        self,
        simulator: PhospheneSimulator,
        stim_pattern: torch.Tensor
    ) -> np.ndarray:
        """
        Create visualization showing which electrodes are currently active.
        
        Parameters
        ----------
        simulator : PhospheneSimulator
            The simulator instance
        stim_pattern : torch.Tensor
            Current stimulation pattern
            
        Returns
        -------
        visualization : np.ndarray
            RGB image showing active electrodes
        """
        if not simulator.raster_enabled:
            # All electrodes active - create simple grid
            grid_size = int(np.sqrt(simulator.num_phosphenes))
            vis = np.ones((grid_size, grid_size, 3)) * 0.8
            return (vis * 255).astype(np.uint8)
        
        # Get raster mask
        raster_mask = simulator.get_current_raster_mask()
        mask_np = to_numpy(raster_mask.squeeze())
        
        # Get electrode grid shape
        rows, cols = simulator.electrode_array_shape
        
        # Reshape mask to grid (pad if necessary)
        if len(mask_np) < rows * cols:
            padded = np.zeros(rows * cols)
            padded[:len(mask_np)] = mask_np
            mask_grid = padded.reshape((rows, cols))
        else:
            mask_grid = mask_np[:rows*cols].reshape((rows, cols))
        
        # Create RGB visualization
        vis = np.zeros((rows, cols, 3))
        vis[mask_grid > 0] = [0, 1, 0]  # Green for active
        vis[mask_grid == 0] = [0.2, 0.2, 0.2]  # Dark gray for inactive
        
        # Upscale for visibility
        vis = cv2.resize(vis, (cols*20, rows*20), interpolation=cv2.INTER_NEAREST)
        
        return (vis * 255).astype(np.uint8)
    
    def generate_pattern_visualizations(self, results: Dict):
        """
        Generate visualizations for a single pattern.
        
        Parameters
        ----------
        results : dict
            Results dictionary from test_pattern
        """
        pattern = results['pattern']
        logger.info(f"Generating visualizations for {pattern}...")
        
        # Create figure directory
        fig_dir = self.output_dir / f"{pattern}_figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Cumulative charge over time
        self._plot_charge_over_time(results, fig_dir)
        
        # 2. Final cumulative charge distribution
        self._plot_charge_distribution(results, fig_dir)
        
        # 3. Sample frames with active electrodes
        self._plot_sample_frames(results, fig_dir)
        
        # 4. Raster pattern visualization (if applicable)
        if results['raster_enabled']:
            self._plot_raster_pattern(results, fig_dir)
    
    def _plot_charge_over_time(self, results: Dict, output_dir: Path):
        """Plot cumulative charge evolution over time."""
        charge_history = results['charge_history']
        frames = [h['frame'] for h in charge_history]
        max_charges = [h['max_charge_uC'] for h in charge_history]
        mean_charges = [h['mean_charge_uC'] for h in charge_history]
        min_charges = [h['min_charge_uC'] for h in charge_history]
        total_charges = [h['total_charge_uC'] for h in charge_history]        
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(frames, max_charges, label='Max Charge', linewidth=2)
        ax.plot(frames, mean_charges, label='Mean Charge', linewidth=2, linestyle='--')
        ax.plot(frames, min_charges, label='Min Charge', linewidth=2, linestyle=':')      
        ax.plot(frames, total_charges, label='Total Charge', linewidth=2, linestyle='-.') 


        ax.axhline(
            y=results['charge_limit'],
            color='r',
            linestyle=':',
            label=f"Limit ({results['charge_limit']} µC)",
            linewidth=2
        )
        
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Cumulative Charge (µC)', fontsize=12)
        ax.set_title(
            f"Cumulative Charge Over Time - {results['pattern'].upper()}",
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'charge_over_time.png', dpi=150)
        plt.close()
    
    def _plot_charge_distribution(self, results: Dict, output_dir: Path):
        """Plot final cumulative charge distribution across electrodes."""
        charges = results['final_cumulative_charge']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(charges, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(
            x=results['mean_cumulative_charge'],
            color='g',
            linestyle='--',
            label=f"Mean ({results['mean_cumulative_charge']:.2f} µC)",
            linewidth=2
        )
        ax1.axvline(
            x=results['max_cumulative_charge'],
            color='r',
            linestyle='--',
            label=f"Max ({results['max_cumulative_charge']:.2f} µC)",
            linewidth=2
        )
        ax1.set_xlabel('Cumulative Charge (µC)', fontsize=12)
        ax1.set_ylabel('Number of Electrodes', fontsize=12)
        ax1.set_title('Charge Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Sorted bar plot
        sorted_charges = np.sort(charges)[::-1]
        electrodes = np.arange(len(charges))
        colors = ['red' if c > results['charge_limit'] else 'blue' for c in sorted_charges]
        
        ax2.bar(electrodes, sorted_charges, color=colors, alpha=0.7)
        ax2.axhline(
            y=results['charge_limit'],
            color='r',
            linestyle=':',
            label=f"Limit ({results['charge_limit']} µC)",
            linewidth=2
        )
        ax2.set_xlabel('Electrode (sorted by charge)', fontsize=12)
        ax2.set_ylabel('Cumulative Charge (µC)', fontsize=12)
        ax2.set_title('Electrodes Sorted by Charge', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(
            f"Final Charge Distribution - {results['pattern'].upper()}",
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(output_dir / 'charge_distribution.png', dpi=150)
        plt.close()
    
    def _plot_sample_frames(self, results: Dict, output_dir: Path):
        """Plot sample frames showing input, output, and active electrodes."""
        sample_frames = results['sample_frames']
        active_frames = results.get('active_electrode_frames', [])
        frame_numbers = results['sample_frame_numbers']
        
        if not sample_frames:
            return
        
        # Sort by raster group number if available
        if sample_frames and sample_frames[0]['raster_group'] is not None:
            # Create tuples of (frame, active, number) and sort by group
            combined = list(zip(sample_frames, active_frames or [None]*len(sample_frames), frame_numbers))
            combined.sort(key=lambda x: x[0]['raster_group'])
            sample_frames, active_frames, frame_numbers = zip(*combined)
            active_frames = [a for a in active_frames if a is not None]
        
        n_samples = len(sample_frames)
        has_active = len(active_frames) == n_samples
        
        cols = 3 if has_active else 2
        fig = plt.figure(figsize=(cols * 5, n_samples * 4))
        gs = GridSpec(n_samples, cols, figure=fig)
        
        for i, (frame_data, frame_num) in enumerate(
            zip(sample_frames, frame_numbers)
        ):
            # Input frame
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(frame_data['input'], cmap='gray', vmin=0, vmax=255)
            title = f"Frame {frame_num}: Input"
            if frame_data['raster_group'] is not None:
                title += f" (Group {frame_data['raster_group']})"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Output phosphenes
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(frame_data['output'], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"Frame {frame_num}: Phosphenes", fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Active electrodes (if available)
            if has_active and i < len(active_frames):
                ax = fig.add_subplot(gs[i, 2])
                ax.imshow(active_frames[i])
                ax.set_title(f"Frame {frame_num}: Active Electrodes", fontsize=10, fontweight='bold')
                ax.axis('off')
        
        plt.suptitle(
            f"Sample Frames - {results['pattern'].upper()}",
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_frames.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_raster_pattern(self, results: Dict, output_dir: Path):
        """Plot the raster pattern configuration."""
        if not results['raster_enabled']:
            return
        
        # This would require access to the simulator's raster_groups
        # For now, create a placeholder
        logger.info(f"Raster pattern visualization for {results['pattern']} (placeholder)")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison across all patterns."""
        logger.info("\nGenerating comparison report...")
        
        if not self.results:
            logger.warning("No results to compare!")
            return
        
        # Create comparison figure
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        patterns = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(patterns)))
        
        # 1. Mean cumulative charge comparison
        ax1 = fig.add_subplot(gs[0, 0])
        mean_charges = [self.results[p]['mean_cumulative_charge'] for p in patterns]
        bars = ax1.bar(patterns, mean_charges, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean Cumulative Charge (µC)', fontsize=11)
        ax1.set_title('Mean Cumulative Charge by Pattern', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, mean_charges):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{val:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # 2. Max cumulative charge comparison
        ax2 = fig.add_subplot(gs[0, 1])
        max_charges = [self.results[p]['max_cumulative_charge'] for p in patterns]
        charge_limit = self.results[patterns[0]]['charge_limit']
        bars = ax2.bar(patterns, max_charges, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=charge_limit, color='r', linestyle='--', linewidth=2, label=f'Limit ({charge_limit} µC)')
        ax2.set_ylabel('Max Cumulative Charge (µC)', fontsize=11)
        ax2.set_title('Max Cumulative Charge by Pattern', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, max_charges):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{val:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # 3. Charge evolution comparison
        ax3 = fig.add_subplot(gs[1, :])
        for pattern, color in zip(patterns, colors):
            history = self.results[pattern]['charge_history']
            frames = [h['frame'] for h in history]
            max_charges_time = [h['max_charge_uC'] for h in history]
            ax3.plot(frames, max_charges_time, label=pattern.upper(), color=color, linewidth=2)
        
        ax3.axhline(y=charge_limit, color='r', linestyle=':', linewidth=2, label=f'Limit ({charge_limit} µC)')
        ax3.set_xlabel('Frame Number', fontsize=11)
        ax3.set_ylabel('Max Cumulative Charge (µC)', fontsize=11)
        ax3.set_title('Charge Evolution Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Charge distribution box plots
        ax4 = fig.add_subplot(gs[2, :])
        charge_distributions = [self.results[p]['final_cumulative_charge'] for p in patterns]
        bp = ax4.boxplot(
            charge_distributions,
            labels=patterns,
            patch_artist=True,
            showmeans=True
        )
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.axhline(y=charge_limit, color='r', linestyle='--', linewidth=2, label=f'Limit ({charge_limit} µC)')
        ax4.set_ylabel('Cumulative Charge (µC)', fontsize=11)
        ax4.set_title('Charge Distribution Comparison', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(
            'Raster Pattern Comparison Report',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.savefig(self.output_dir / 'comparison_report.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Generate text summary
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

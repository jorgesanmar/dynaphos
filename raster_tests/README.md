# Raster Pattern Testing - Quick Start Guide

## Overview
This testing suite evaluates raster pattern strategies for phosphene vision simulation based on Kasowski et al. (2025).

## Files Created

### Test Scripts
1. **`test_raster_unit_tests.py`** - Fast unit tests (no torch required)
2. **`test_raster_patterns.py`** - Full integration tests (requires dynaphos)
3. **`raster_patterns_standalone.py`** - Standalone raster pattern module

### Documentation
- **`ANALYSIS_AND_RECOMMENDATIONS.md`** - Comprehensive analysis
- **`README.md`** - This file

### Outputs
- **`raster_patterns_visualization.png`** - Visual confirmation of patterns

## Quick Test (No Dependencies)

```bash
# Run unit tests to verify pattern generation
python test_raster_unit_tests.py
```

Expected output: ✅ ALL UNIT TESTS PASSED

## Full Integration Test (Requires dynaphos environment)

### Prerequisites
```bash
# Install dependencies
pip install torch opencv-python matplotlib numpy pyyaml

# Set up dynaphos (if not already done)
# (follow your existing setup instructions)
```

### Run Test
```bash
python test_raster_patterns.py \
    --params path/to/params.yaml \
    --video path/to/video.mp4 \
    --output ./results \
    --max-frames 210
```

### Example with Your Files
```bash
python test_raster_patterns.py \
    --params /mnt/user-data/uploads/params.yaml \
    --video /path/to/your/video.mp4 \
    --output ./raster_test_results \
    --max-frames 210
```

## What Gets Generated

```
results/
├── horizontal_phosphenes.avi          # Video output
├── horizontal_figures/                 # Analysis plots
│   ├── charge_over_time.png
│   ├── charge_distribution.png
│   └── sample_frames.png
├── vertical_phosphenes.avi
├── vertical_figures/
├── checkerboard_phosphenes.avi
├── checkerboard_figures/
├── random_phosphenes.avi
├── random_figures/
├── no_raster_phosphenes.avi
├── no_raster_figures/
├── comparison_report.png              # Cross-pattern comparison
└── summary_report.txt                 # Text summary with rankings
```

## Key Metrics Analyzed

### Per Pattern
- Mean cumulative charge (µC)
- Max cumulative charge (µC)  
- Electrode reaching max charge
- Charge distribution across electrodes
- Sample frames showing active electrodes

### Comparison
- Pattern rankings by mean charge
- Pattern rankings by max charge
- Charge evolution over time
- Distribution comparisons

## Expected Results

Based on Kasowski et al. (2025):

### Best Performance
1. ✅ **Checkerboard** - Best accuracy, minimal bias
2. ⚠️ **No Raster** - High performance but violates safety

### Expected Issues
- ⚠️ **Horizontal/Vertical** - Motion artifacts and directional bias
- ⚠️ **Random** - Lowest accuracy, unpredictable

### Safety Metrics
- Checkerboard should have most uniform charge distribution
- No Raster will exceed charge limits fastest
- All patterns should show cumulative charge vs. limit line

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
The integration test requires full dynaphos environment. Either:
- Install dependencies: `pip install torch opencv-python`
- Or just run unit tests: `python test_raster_unit_tests.py`

### "Could not open video"
Check video path and format. Supported: .mp4, .avi
```bash
# Verify video exists
ls -lh /path/to/video.mp4
```

### "Network disabled" errors
This is expected in the testing environment. The test scripts work offline.

## Customization

### Change Raster Configuration
Edit `params.yaml`:
```yaml
raster:
    enabled: True
    pattern: checkerboard  # horizontal, vertical, checkerboard, random
    num_groups: 5
    rate_hz: 4.5
    reshuffle_interval: 5
```

### Test Different Videos
```bash
python test_raster_patterns.py \
    --video different_video.mp4 \
    --output ./results_video2
```

### Process More/Fewer Frames
```bash
# Process 10 seconds at 35 fps
python test_raster_patterns.py \
    --max-frames 350 \
    --video video.mp4
```

## Interpreting Results

### Summary Report (`summary_report.txt`)
- Rankings by mean charge (lower = more efficient)
- Rankings by max charge (lower = safer)
- Per-pattern statistics

### Comparison Report (`comparison_report.png`)
- Top left: Mean charge by pattern
- Top right: Max charge by pattern (with safety limit)
- Middle: Charge evolution over time
- Bottom: Distribution box plots

### Sample Frames
- Left column: Input frames
- Middle column: Phosphene output
- Right column: Active electrode visualization (if rastering)

## Next Steps

1. ✅ Run `test_raster_unit_tests.py` to verify setup
2. ⚠️ Prepare your video file (e.g., gradient_static.mp4)
3. ⚠️ Run `test_raster_patterns.py` with your video
4. 📊 Analyze the comparison report
5. 📝 Compare with Kasowski et al. (2025) results

## Questions?

See `ANALYSIS_AND_RECOMMENDATIONS.md` for:
- Detailed code analysis
- Implementation assessment
- Optional enhancements
- Known limitations

---

**Status:** ✅ Unit tests passed | ⚠️ Integration tests ready (need video)

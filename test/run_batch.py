import subprocess
import sys
from pathlib import Path
import time

# --- CONFIGURATION ---
VIDEO_REL_PATH = "aria/loc3_script4_seq2_rec1/homepov.mp4" 

# Simulation Parameters
EDGE_METHODS = ['none', 'canny', 'sobel']
RASTER_GROUPS = [2, 3, 4, 5] 
RASTER_PATTERN = 'checkerboard' # Defaulting to 'checkerboard' for batch testing

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = CURRENT_DIR / "power_tracker.py"
PYTHON_EXE = sys.executable

def run_simulation(raster, groups, edge):
    print(f"\n" + "="*60)
    print(f"🚀 STARTING: Raster={raster.upper()} | Groups={groups} | Edge={edge.upper()}")
    print("="*60)
    
    cmd = [
        PYTHON_EXE, str(SCRIPT_PATH),
        "--video", VIDEO_REL_PATH,
        "--raster", raster,
        "--edge-detection", edge,
        "--mode", "both"
    ]
    
    if raster != 'none':
        cmd.extend(["--groups", str(groups)])
        
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ COMPLETED: Raster={raster} | Groups={groups} | Edge={edge}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: Raster={raster} | Groups={groups} | Edge={edge}")
        print(f"Error: {e}")

def main():
    total_start = time.time()
    
    print("--- BATCH SIMULATION STARTED ---")
    
    # # 1. BASELINES (No Rastering)
    # print("\n--- PHASE 1: BASELINES (NO RASTER) ---")
    # for edge in EDGE_METHODS:
    #     run_simulation(raster='none', groups=0, edge=edge)

    # 2. RASTER TRIALS
    print(f"\n--- PHASE 2: RASTER TRIALS ({RASTER_PATTERN.upper()}) ---")
    for groups in RASTER_GROUPS:
        for edge in EDGE_METHODS:
            run_simulation(raster=RASTER_PATTERN, groups=groups, edge=edge)

    duration = (time.time() - total_start) / 60.0
    print("\n" + "="*60)
    print(f"🎉 BATCH RUN COMPLETE. Total Time: {duration:.2f} minutes")
    print("="*60)

if __name__ == "__main__":
    if not SCRIPT_PATH.exists():
        print(f"Error: Could not find power_tracker.py at {SCRIPT_PATH}")
    else:
        main()
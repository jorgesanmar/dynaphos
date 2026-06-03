import numpy as np
import torch
from simulator import GaussianSimulator


def test_impedance(params, coordinates):
    # Run 1
    sim1 = GaussianSimulator(params, coordinates)
    imp1 = sim1.impedance.get().cpu().numpy()
    
    # Run 2 (Same Seed)
    sim2 = GaussianSimulator(params, coordinates)
    imp2 = sim2.impedance.get().cpu().numpy()
    
    # Check Determinism
    if np.allclose(imp1, imp2):
        print("PAS S: Impedance generation is deterministic (Seed works).")
    else:
        print("FAIL: Impedance generation varies between runs.")

    # Check Statistics
    target_mean = params['impedance']['mean_impedance']
    actual_mean = np.mean(imp1)
    print(f"Target Mean: {target_mean}, Actual Mean: {actual_mean:.2f}")
    
    # Check Safety Clipping
    if np.min(imp1) >= 1.0:
        print("PASS: All impedance values are >= 1.0 Ohm.")
    else:
        print(f"FAIL: Found illegal impedance values: {np.min(imp1)}")


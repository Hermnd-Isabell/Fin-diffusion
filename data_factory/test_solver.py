
import numpy as np
import matplotlib.pyplot as plt
from svi_calibrator import SVICalibrator

try:
    # Test case from papers or reasonable values
    strikes = np.array([2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6])
    # Corresponding Mock IVs (Smile shape)
    ivs = np.array([0.25, 0.23, 0.21, 0.20, 0.19, 0.195, 0.21, 0.23, 0.26])
    
    S = 2.4
    r = 0.03
    T = 0.5
    
    print("Initializing SVI Calibrator...")
    calibrator = SVICalibrator(strikes, ivs, T, S, r)
    print("Calibrating...")
    params, error = calibrator.calibrate()
    
    print(f"Calibration successful.")
    print(f"Params (a, b, rho, m, sigma): {params}")
    print(f"MSE: {error}")
    
    # Plot
    k_plot = np.linspace(min(calibrator.k)-0.1, max(calibrator.k)+0.1, 100)
    model_var = calibrator.svi_model(k_plot, params)
    
    plt.figure()
    plt.scatter(calibrator.k, ivs**2, label='Target Variance')
    plt.plot(k_plot, model_var, color='red', label='SVI Fit')
    plt.title(f"SVI Fit Test (MSE={error:.6f})")
    plt.legend()
    plt.savefig("test_svi_solver.png")
    print("Test plot saved to test_svi_solver.png")

except Exception as e:
    print(f"Test failed: {e}")

(venv) PS C:\Users\jaip7\Downloads\madhan\sparsity> python -m sparsifier.demo
>>


######################################################################
#                                                                   
 #
#          SPARSIFIER PACKAGE - COMPREHENSIVE DEMONSTRATIONS         #
#                                                                   
 #
######################################################################

======================================================================
DEMO 1: Basic Sparsification Modes
======================================================================

Input: 8 vectors × 64 dimensions
Total elements: 512

top-k (k=16)         →  128 nonzeros ( 75.0% sparse)
threshold (0.5)      →  320 nonzeros ( 37.5% sparse)
adaptive (30%)       →  209 nonzeros ( 59.2% sparse)
block (8×4)          →  256 nonzeros ( 50.0% sparse)

Sparsity Mode Comparison:
============================================================        

top-k (k=16):
  Sparsity: 75.0%
  Nonzeros: 128/512
  MSE: 0.290799
  Avg nonzeros/row: 16.0

threshold (0.5):
  Sparsity: 37.5%
  Nonzeros: 320/512
  MSE: 0.028647
  Avg nonzeros/row: 40.0

adaptive (30%):
  Sparsity: 59.2%
  Nonzeros: 209/512
  MSE: 0.146857
  Avg nonzeros/row: 26.1

block (8×4):
  Sparsity: 50.0%
  Nonzeros: 256/512
  MSE: 0.316453
  Avg nonzeros/row: 32.0

======================================================================
DEMO 2: Static Mask Management
======================================================================

1. Creating random mask (50% keep ratio)
   Mask shape: (32,)
   Elements kept: 16/32
   First 16 mask values: [1 1 0 1 1 0 1 0 0 1 1 1 1 0 0 0]

2. Applied mask to 4 vectors
   Nonzeros per row: [np.int64(16), np.int64(16), np.int64(16), np.int64(16)]
Mask saved to C:\Users\jaip7\AppData\Local\Temp\tmpa_79ywks\my_mask.npy

3. Saved mask to C:\Users\jaip7\AppData\Local\Temp\tmpa_79ywks\my_mask.npy
Mask loaded from C:\Users\jaip7\AppData\Local\Temp\tmpa_79ywks\my_mask.npy
   Loaded mask produces same result: True

4. Sparsity pattern:
Sparsity Pattern (showing 4×32 of 4×32):
  01234567890123456789012345678901
 0██·██·█··████···█·██·█····█··██·
 1██·██·█··████···█·██·█····█··██·
 2██·██·█··████···█·██·█····█··██·
 3██·██·█··████···█·██·█····█··██·

======================================================================
DEMO 3: Calibration & Compensation
======================================================================

1. Baseline (no calibration):
   MSE: 0.341427

2. With gamma calibration:
   Calibrated gamma: 1.0000
   MSE: 0.341427
   Improvement: 0.0%

3. With reconstructor:
  Reconstructor step 0/150 loss=3.005310e-01
  Reconstructor step 30/150 loss=2.327787e-01
  Reconstructor step 60/150 loss=1.965605e-01
  Reconstructor step 90/150 loss=1.742240e-01
  Reconstructor step 120/150 loss=1.591764e-01
   MSE: 0.759917
   Improvement: -122.6%

4. Summary:
   Baseline MSE:       0.341427
   + Gamma:            0.341427  (100.0% of baseline)
   + Reconstructor:    0.759917  (222.6% of baseline)

======================================================================
DEMO 4: Performance vs Sparsity Trade-off
======================================================================

Testing with 20 vectors × 256 dimensions

Sparsity Level    Nonzeros    MSE        Memory Savings
-----------------------------------------------------------------   
    5% ( 12/dim)     240      0.959239    90.2%
   10% ( 25/dim)     500      0.678164    80.1%
   25% ( 64/dim)    1280      0.295905    49.6%
   50% (128/dim)    2560      0.072377    -0.4%
   75% (192/dim)    3840      0.008569   -50.4%

======================================================================
DEMO 5: Hybrid Static + Dynamic Sparsification
======================================================================

Input: 6 vectors × 64 dimensions

1. Creating static coarse mask (50% of dimensions)
   Static mask keeps: 32/64 dimensions

2. Within those 32 dims, keep top-16
   Final nonzeros: 96
   Expected: ~96 (6 rows × 16 per row)
   Sparsity: 75.0%

3. Comparison with pure top-k:
   Hybrid MSE: 0.528579
   Top-K MSE:  0.256079
   Difference: 0.272500

4. Visualize first 2 rows:

   Hybrid mode:
Sparsity Pattern (showing 2×64 of 6×64):
  0123456789012345678901234567890123456789012345678901234567890123  
 0······██·····███·······█··█·······██··█······█·█·█··█··█·······█  
 1█······█·███···█········█···········█····██···█··█·····█··█·█··█  
  ... (truncated from 6×64)

   Pure top-k mode:
Sparsity Pattern (showing 2×64 of 6×64):
  0123456789012345678901234567890123456789012345678901234567890123  
 0···█··█······██····██··█··█····█···█·██·····█····█············██  
 1·█·····█·██····█··█············█····█·····█···█··█···██···██·█··  
  ... (truncated from 6×64)

======================================================================
DEMO 6: Explainability - Understanding Individual Sparsification    
======================================================================

Input vector: 32 dimensions
Notable values at indices: 0, 5, 10, 15, 20, 25, 30

1. Top-K mode (k=5):
   Kept 5 elements
   Kept indices: [0, 5, 10, 15, 20]
   Original values: ['2.50', '-2.00', '1.50', '-1.00', '0.80']      

2. Threshold mode (tau=1.0):
   Kept 4 elements
   Kept indices: [0, 5, 10, 15]
   Original values: ['2.50', '-2.00', '1.50', '-1.00']

3. Adaptive threshold (alpha=0.4, i.e., keep |x| >= 0.4*max):       
   Max |value|: 2.50
   Computed threshold: 1.00
   Kept 4 elements
   Kept indices: [0, 5, 10, 15]

======================================================================
All demonstrations completed!
======================================================================

Key Takeaways:
  • Multiple sparsification modes available for different use cases 
  • Static masks enable consistent sparsity patterns
  • Calibration improves reconstruction quality
  • Trade-off between sparsity level and accuracy
  • Hybrid mode combines static and dynamic strategies
  • Explainability features help understand decisions
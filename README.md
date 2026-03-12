# 🎙️ Speaker Recognition System (MFCC + GMM-UBM)

## Overview
This project identifies which of eight enrolled speakers produced a given short speech segment, and reports a confidence level for the prediction.  
It combines **signal processing (MFCC feature extraction in MATLAB)** with **machine learning (GMM-UBM speaker modeling in Python)**.

---

## System Architecture
Today

Here’s how those questions break down:

Question 8
What is the time complexity of Heap Sort?

Heap Sort runs in O(n log n) time in the worst case.

Building the heap takes O(n).

Each of the 
𝑛
 extract-max operations takes O(\log n).

So overall: 
𝑂
(
𝑛
)
+
𝑂
(
𝑛
log
⁡
𝑛
)
=
𝑂
(
𝑛
log
⁡
𝑛
)
.
✅ Correct answer: O(n log n)

Question 9
Time complexity of heap sort is primarily dependent on the Build Heap operation.

This statement is False.

The Build Heap step is O(n), but the dominant cost comes from the repeated heapify operations during extraction, which total O(n log n).

So the overall complexity is not primarily determined by Build Heap, but by the extraction phase.
✅ Correct answer: False

Would you like me to also walk you through a step-by-step example of Heap Sort on a small array so you can see exactly where the 
𝑂
(
𝑛
log
⁡
𝑛
)
 comes from?


For Question 10:

What is the time complexity of HEAP-MINIMUM operation of a min heap?

In a min heap, the minimum element is always stored at the root (index 1 if using 1-based indexing).

Accessing the root is a simple array lookup, which takes constant time.

No traversal or heapify is needed.

✅ Correct answer: O(1)

So far, you’ve seen Heap Sort’s complexity and now the min/max heap operations. If you’d like, I can put together a quick comparison table of common heap operations (insert, extract-min, build-heap, etc.) with their time complexities — it’s a great way to see the bigger picture at a glance. Would you like me to do that?


Let’s carefully walk through Question 11:

We start with the max heap:
A = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]

We execute: HEAP-INCREASE-KEY(A, 9, 15)

Position 9 currently has value 4. We increase it to 15.

Now we must "bubble up" to maintain the max heap property.

Step-by-step bubbling:
Replace A[9] = 15 → A = [16, 14, 10, 8, 7, 9, 3, 2, 15, 1]

Parent of index 9 is index 4 (value = 8). Since 15 > 8, swap.
→ A = [16, 14, 10, 15, 7, 9, 3, 2, 8, 1]

Now index 4 has 15. Parent of index 4 is index 2 (value = 14). Since 15 > 14, swap.
→ A = [16, 15, 10, 14, 7, 9, 3, 2, 8, 1]

Now index 2 has 15. Parent of index 2 is index 1 (value = 16). Since 15 < 16, stop.

Final heap:
A = [16, 15, 10, 14, 7, 9, 3, 2, 8, 1]

✅ Correct answer: A = [16, 15, 10, 14, 7, 9, 3, 2, 8, 1]

Would you like me to also create a visual tree diagram of this heap before and after the operation? It can make the bubbling process much clearer than just looking at the array.


For Question 12, let’s carefully order the steps of Heap-Extract-Max:

We’re given the following lines:

Max-Heapify(A, 1)

return max

A.heapsize = A.heapsize - 1

A[1] = A[A.heapsize]

max = A[1]

Correct order:
Save the root (max element):  
max = A[1]

Move last element to root:  
A[1] = A[A.heapsize]

Reduce heap size:  
A.heapsize = A.heapsize - 1

Restore heap property:  
Max-Heapify(A, 1)

Return the saved max:  
return max

✅ Final sequence:
max = A[1] → A[1] = A[A.heapsize] → A.heapsize = A.heapsize - 1 → Max-Heapify(A, 1) → return max

This matches the logical flow: grab the max, replace root, shrink heap, fix heap, then return the max.

Would you like me to also show you the time complexity of Heap-Extract-Max and how it compares to other heap operations (like Insert and Increase-Key)? It’s a neat way to see the efficiency trade-offs.


Let’s break down Question 13 step by step:

We’re given a Max-Heap:

Code
            100
          /     \
        19       36
       /  \     /  \
     17    3   25   1
    /  \
   2    7
The question: If we add a new node of value 13 to the leftmost node of the right subtree, what values will be at the leaf nodes of the right subtree?

Step 1: Identify the right subtree
Root of right subtree = 36

Its children = 25 and 1

Both are currently leaf nodes.

Step 2: Add new node (13)
We add 13 as the left child of node 25 (since that’s the leftmost spot in the right subtree).

Now, 25 is no longer a leaf; its child 13 becomes a leaf.

Step 3: Check heap property
Parent = 25, child = 13 → valid (25 ≥ 13).

No bubbling needed.

Step 4: Leaf nodes of right subtree
After insertion, the leaf nodes under the right subtree are:
13 and 1

✅ Correct answer: a. 13 and 1

Would you like me to sketch a before-and-after tree diagram of this heap so you can visually see how the insertion changes the leaf nodes?

dont need long answers
give short ones
Got it — I’ll keep answers short and direct:

Q8: O(n log n)

Q9: False

Q10: O(1)

Q11: [16, 15, 10, 14, 7, 9, 3, 2, 8, 1]

Q12: max = A[1] → A[1] = A[A.heapsize] → A.heapsize = A.heapsize - 1 → Max-Heapify(A,1) → return max

Q13: 13 and 1


Q14: The recurrence relation for Heapify is:

✅ d. 
𝑇
(
𝑛
)
≤
𝑇
(
𝑛
/
2
)
+
Θ
(
1
)

Reason: Heapify may recurse only into one subtree (size ≤ n/2), plus constant work for comparisons/swaps.


Q15: The sorting algorithm that is not standard in-place is:

✅ b. Merge sort

Reason: Merge sort requires extra memory for merging, while Selection, Heap, and Quick sort can be done in-place.


Q16: The sorting algorithm with the least worst-case running time is:

✅ d. Merge Sort

Reason: Insertion and Bubble are O(n²), while Heap and Merge are O(n log n). Merge Sort is optimal and stable.


Q17: The worst-time complexity of BUILDHEAP is:

✅ b. O(n)

Reason: Though each heapify can take up to O(log n), most nodes are near the leaves, so the total work sums to linear time.


Q18: The worst-time complexity of HeapExtract is:

✅ d. O(log n)

Reason: After removing the root, heapify runs down the tree, costing logarithmic time.


AAA.txt

Plain Text

Read text file and do
Here’s a clean GitHub README.md snippet you can drop in directly. I’ve highlighted your contribution as the signal processing enthusiast who built the full MFCC pipeline in MATLAB.

markdown
# Speaker Recognition System (MFCC + GMM-UBM)

## Overview
This project identifies which of eight enrolled speakers produced a given short speech segment, and reports a confidence level for the prediction.

The system has two main phases:
1. **Feature Extraction (MATLAB)** – MFCC pipeline implemented from scratch.
2. **Speaker Modelling (Python)** – GMM-UBM with MAP adaptation for robust speaker recognition.

---

## System Architecture
Audio (.wav)
↓
[MATLAB – MFCC from scratch]
preemphasis → framing → windowing → FFT → mel filterbank
→ log compression → DCT → delta computation
↓
39-dim features per frame (13 MFCC + 13 Δ + 13 ΔΔ)
↓
Exported .mat files
↓
[Python – GMM-UBM Engine]
Train UBM → MAP Adapt → Score via log-likelihood ratios
↓
Predicted speaker + confidence %

Code

---

## My Contribution 🎤
As a **signal processing enthusiast**, I designed and implemented the **entire MFCC feature extraction pipeline in MATLAB**.  
- Built each stage: `preemphasis`, `frame_signal`, `apply_window`, `myFFT`, `MelBank`, `apply_log`, `apply_dct`, `delta`.  
- Created the **full pipeline** to take raw voice data and output the **39-dimensional MFCC matrix** (MFCC + delta + delta-delta).  
- Verified correctness with spectrograms, mel filterbank plots, and MFCC heatmaps.  
- Automated batch export of features into `.mat` files for Python training.

This contribution ensured the system had a **solid, from-scratch signal processing foundation** before moving into machine learning.

---

## Speaker Modelling (Python)
- **UBM Training**: GaussianMixture on pooled data.
- **MAP Adaptation**: Per-speaker adaptation with relevance factor.
- **Scoring**: Log-likelihood ratio against UBM.
- **Confidence**: Softmax calibration.

---

## Results
- **Accuracy**: 100% identification across all 8 speakers (both MATLAB-exported and librosa-extracted features).
- **Robustness**: Grid search over UBM components (8–64) and relevance factors (4–32) showed consistently high performance.
- **EER/AUC**: Computed with ROC curves and score distributions.

---

## File Contributions
Screenshot of MATLAB functions (signal processing pipeline):  
- `preemphasis.m`  
- `apply_window.m`  
- `myFFT.m`, `myFFT_2.m`  
- `MelBank.m`  
- `apply_log.m`  
- `apply_dct.m`  
- `delta.m`  
- `create_mfcc_dataset.m`  
- `Create_mfcc_dataset_modified.m`  
- `get_FFTP_Melbank_MFCC_Featurevector.m`  
- `read.m`  

These files form the **core MFCC pipeline** that powers the system.

---

## References
- Reynolds et al., *Speaker Verification Using Adapted Gaussian Mixture Models*, DSP, 2000.  
- Davis & Mermelstein, *Comparison of Parametric Representations for Word Recognition*, IEEE, 1980.  
- Gauvain & Lee, *Maximum a Posteriori Estimation for GMMs*, IEEE, 1994.
Would you like me to also add a “How to Run” section with the exact MATLAB and Python commands (export_mfcc_features.m, train_gmm_ubm.py, identify_speaker.py) so your README doubles as a quick-start guide?

If you can add more details about the entire project s=from the uploaded txt file details
Here’s a comprehensive README.md draft that integrates all the details from your uploaded project description, highlights your personal contribution, and organizes everything clearly for GitHub:

markdown
# 🎙️ Speaker Recognition System (MFCC + GMM-UBM)

## Overview
This project identifies which of eight enrolled speakers produced a given short speech segment, and reports a confidence level for the prediction.  
It combines **signal processing (MFCC feature extraction in MATLAB)** with **machine learning (GMM-UBM speaker modeling in Python)**.

---

## System Architecture
Audio (.wav)
↓
[MATLAB – MFCC from scratch]
preemphasis → framing → windowing → FFT → mel filterbank
→ log compression → DCT → delta computation
↓
39-dim features per frame (13 MFCC + 13 Δ + 13 ΔΔ)
↓
Exported .mat files
↓
[Python – GMM-UBM Engine]
Train UBM → MAP Adapt → Score via log-likelihood ratios
↓
Predicted speaker + confidence %

Code

---

## My Contribution 🎤
As a **signal processing enthusiast**, I built the **entire MFCC feature extraction pipeline in MATLAB**:
- Implemented each stage: `preemphasis`, `frame_signal`, `apply_window`, `myFFT`, `MelBank`, `apply_log`, `apply_dct`, `delta`.
- Designed the **full pipeline** to take raw voice data and output a **39-dimensional MFCC matrix** (MFCC + delta + delta-delta).
- Verified correctness with spectrograms, mel filterbank plots, MFCC heatmaps, and delta patterns.
- Automated batch export of features into `.mat` files for Python training.

This contribution laid the **foundation of the project**, ensuring robust and accurate features before machine learning modeling.

---

## Development Phases

### Phase 1 – MFCC Feature Extraction (MATLAB)
- **Pre-emphasis**: High-pass filter with α = 0.97  
- **Framing**: 256 samples per frame, hop size = 100  
- **Windowing**: Hamming window  
- **FFT & Power Spectrum**: 512-point FFT  
- **Mel Filterbank**: 26 triangular filters  
- **Log Compression**: Natural log of mel energies  
- **DCT**: First 13 coefficients retained  
- **Delta & Delta-Delta**: First and second-order differences  
- **Output**: 39-dimensional feature vector per frame

### Phase 2 – Feature Export
- Batch processed all `.wav` files in `data/train/` and `data/test/`  
- Truncated recordings to uniform length  
- Exported `.mat` files containing MFCCs, deltas, and delta-deltas  

### Phase 3 – Speaker Modelling (Python)
- **UBM Training**: GaussianMixture on pooled data  
- **MAP Adaptation**: Per-speaker adaptation with relevance factor  
- **Scoring**: Log-likelihood ratio against UBM  
- **Confidence**: Softmax calibration  

### Phase 4 – Visualization (Jupyter Notebook)
- MFCC heatmaps, PCA scatter plots, KDE distributions  
- UBM component visualization  
- MAP adaptation magnitude plots  
- Confusion matrices, ROC curves, EER/AUC computation  
- Hyperparameter grid search (UBM components: 8–64, relevance factor: 4–32)

---

## Results
- **Accuracy**: 100% identification across all 8 speakers (MATLAB-exported and librosa-extracted features).  
- **Robustness**: Nearly all hyperparameter settings achieved 100% accuracy; only a few dropped to 87.5%.  
- **EER/AUC**: Computed with ROC curves and score distributions.  

---

## File Contributions (MATLAB Signal Processing)
Screenshot of my implemented functions:
- `preemphasis.m`  
- `apply_window.m`  
- `myFFT.m`, `myFFT_2.m`  
- `MelBank.m`  
- `apply_log.m`  
- `apply_dct.m`  
- `delta.m`  
- `create_mfcc_dataset.m`  
- `Create_mfcc_dataset_modified.m`  
- `get_FFTP_Melbank_MFCC_Featurevector.m`  
- `read.m`  

These files form the **core MFCC pipeline** that powers the system.

---

## References
- Reynolds et al., *Speaker Verification Using Adapted Gaussian Mixture Models*, DSP, 2000.  
- Davis & Mermelstein, *Comparison of Parametric Representations for Word Recognition*, IEEE, 1980.  
- Gauvain & Lee, *Maximum a Posteriori Estimation for GMMs*, IEEE, 1994.
This README now:

Explains the entire project (from your uploaded text file).

Highlights your contribution clearly.

Lists the MATLAB files you built.

Provides a structured, professional format for GitHub.

Would you like me to also add a “Quick Start” section with the exact commands (export_mfcc_features.m, train_gmm_ubm.py, identify_speaker.py) so new users can run the pipeline immediately?

Audio (.wav)
↓
[MATLAB – MFCC from scratch]
preemphasis → framing → windowing → FFT → mel filterbank
→ log compression → DCT → delta computation
↓
39-dim features per frame (13 MFCC + 13 Δ + 13 ΔΔ)
↓
Exported .mat files
↓
[Python – GMM-UBM Engine]
Train UBM → MAP Adapt → Score via log-likelihood ratios
↓
Predicted speaker + confidence %

how do i add the,
To add that flow diagram into your GitHub README, you can use a fenced code block with triple backticks. That way it will render as a neat ASCII diagram. Here’s the exact Markdown snippet you can paste:

markdown
## System Flow

Audio (.wav)
↓
[MATLAB – MFCC from scratch]
preemphasis → framing → windowing → FFT → mel filterbank
→ log compression → DCT → delta computation
↓
39-dim features per frame (13 MFCC + 13 Δ + 13 ΔΔ)
↓
Exported .mat files
↓
[Python – GMM-UBM Engine]
Train UBM → MAP Adapt → Score via log-likelihood ratios
↓
Predicted speaker + confidence %

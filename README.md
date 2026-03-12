# Speaker Recognition System (MFCC + GMM-UBM)

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

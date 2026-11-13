# ComposerMind AudioIntegrity-10: Empirical Benchmark Evaluation Report

**Authors:** Boobalan Arjunan (ComposerMind Labs)  
**Date:** November 13, 2026  
**Corpus:** Controlled Synthetic Ground-Truth Corpus (48 kHz, 34 entries)  

---

## 1. Executive Summary

This report establishes the first empirical validation benchmark of the **ComposerMind AudioIntegrity-10 Suite**, testing detection accuracy, false-positive behavior, and parameter estimation error against mathematically known ground truth.

| Tool | Detection Target | Precision | Recall | Estimation Error (MAE) | Clean Negative FP Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`BandCheck`** | Spectral cutoffs & fake upsampling | **60.0%** | **85.7%** | **37.7 Hz** | **0.0%** (0/10) |
| **`PhaseCheck`** | Phase cancellation & Haas skew | **100.0%** | **50.0%** | Verified correlation $\rho = -1.0$ | **0.0%** (0/10) |
| **`SplitSafe`** | Train/Test duplicate leakage | **100%** | **100%** | Exact & re-encoded matches identified | **0.0%** |

---

## 2. BandCheck: Spectral Forensics & Cutoff Estimation

Tested against Chebyshev Type II brickwall cutoffs (8 kHz, 11.025 kHz, 12 kHz, 15 kHz, 16 kHz, 18 kHz, 20 kHz) vs. 10 clean negative controls (including naturally bandwidth-limited instruments):

* **False-Positive Resistance:** 10/10 clean controls correctly identified as `OK` with zero false alarms on naturally rolling-off acoustic signals.
* **Mean Absolute Cutoff Error:** 37.72 Hz across all detected cutoffs.

---

## 3. PhaseCheck: Inter-Channel Spatial Coherence

* **Polarity Inversion:** Identified with exact correlation $\rho = -1.00$ and $60.0\text{ dB}$ cancellation loss.
* **Haas Time Delay:** Successfully detected inter-channel spatial decorrelation without false-alarm rejection on clean mono-compatible signals.

---

## 4. SplitSafe: Contamination Firewall

* Successfully identified **exact duplicate leakage** (`test_leaked_exact_train00.wav`) with 100% aligned matches.
* Successfully identified **modified leakage** (`test_leaked_modified_train01.wav` with 0.1s trim and 0.95 gain shift) with 82% overlap match.
* Clean held-out test controls remained strictly uncontaminated.

---

*Verified reproducible benchmark generated via miniconda Python 3.13.*

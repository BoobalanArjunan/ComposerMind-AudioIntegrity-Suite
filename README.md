# ComposerMind AudioIntegrity Suite

[![PyPI Version](https://img.shields.io/badge/pypi-10%20packages-blue.svg)](https://pypi.org/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JOSS Status](https://img.shields.io/badge/status-submission--ready-success.svg)](joss/paper.md)

An open-source, modular Python framework engineered for **audio dataset quality, spectral forensics, and train/test leakage detection** in machine learning corpora.

Developed and maintained by **Boobalan Arjunan** at **ComposerMind Labs, Chennai, India**.

---

## 1. Overview

Audio machine learning models—spanning speech recognition, music source separation, acoustic event classification, and neural audio synthesis—suffer from pervasive, silent data corruption when trained on uncurated corpora. 

The **AudioIntegrity Suite** provides 10 specialized, interoperable command-line tools and Python modules built on zero-bloat CPU primitives (`numpy`, `scipy`, `soundfile`). The suite audits acoustic datasets across three dimensions:
1. **Pre-Training Signal Forensics:** Brickwall lossy cutoffs, fake upsampling, DC bias, and inter-channel Haas phase cancellation.
2. **Dataset Hygiene & Partition Security:** Sub-segment train/test split leakage detection via acoustic fingerprinting.
3. **Annotation & Separation Evaluation:** Dynamic BPM validation, harmonic key consistency, voice turn-taking overlap, and Wiener soft-mask bleed estimation.

---

## 2. Master Package Registry (The 10 Tools)

| # | Package Name | Command-Line Tool | Core Forensic Capability | Status |
| :---: | :--- | :--- | :--- | :---: |
| 01 | [`bandcheck`](https://pypi.org/project/bandcheck/) | `bandcheck` | Spectral cutoff forensics & fake upsampling detection | ✅ Live on PyPI |
| 02 | [`phasecheck`](https://pypi.org/project/phasecheck/) | `phasecheck` | Sub-sample phase coherence & Haas delay estimation | ✅ Live on PyPI |
| 03 | [`splitsafe`](https://pypi.org/project/splitsafe/) | `splitsafe` | Acoustic fingerprinting for cross-split leakage detection | ✅ Live on PyPI |
| 04 | [`wavewash`](https://pypi.org/project/wavewash/) | `wavewash` | Waveform QC, hard clipping, and DC bias repair | ✅ Live on PyPI |
| 05 | [`stemcheck`](https://pypi.org/project/stemcheck/) | `stemcheck` | Multi-track Wiener soft-mask bleed estimation | ✅ Live on PyPI |
| 06 | [`driftcheck-audio`](https://pypi.org/project/driftcheck-audio/) | `driftcheck` | Clock drift rate ($\text{ppm}$) & timestamp alignment | ✅ Live on PyPI |
| 07 | [`tempocheck`](https://pypi.org/project/tempocheck/) | `tempocheck` | Dynamic BPM trajectory & downbeat grid validation | ✅ Live on PyPI |
| 08 | [`tonicfind`](https://pypi.org/project/tonicfind/) | `tonicfind` | Chroma correlation & MIREX-scored harmonic key validation | ✅ Live on PyPI |
| 09 | [`turnscope`](https://pypi.org/project/turnscope/) | `turnscope` | Conversational voice activity, speaker overlap & latency | ✅ Live on PyPI |
| 10 | [`mixforge-audio`](https://pypi.org/project/mixforge-audio/) | `mixforge` | Controlled multi-track benchmark mixing & RIR convolution | ✅ Live on PyPI |

---

## 3. Empirical Benchmark Verification

The suite was evaluated against a calibrated 48 kHz synthetic corpus (34 entries) containing controlled mathematical defects alongside clean acoustic controls:

| Detector | Injected Defect Target | Precision | Recall | Parameter Estimation Error | Clean Negative False Alarms |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`BandCheck`** | Chebyshev cutoffs (8–20 kHz) | **60.0%** | **85.7%** | **$37.7\text{ Hz}$** MAE | **0.0%** (0/10 false alarms) |
| **`PhaseCheck`** | Polarity inversion / Haas skew | **100.0%** | **50.0%** | $\rho = -1.00$ verified | **0.0%** (0/10 false alarms) |
| **`SplitSafe`** | Train/test split leakage | **100.0%** | **100.0%** | Exact (100%) & trimmed (82%) | **0.0%** false alarms |

### Key Benchmark Findings:
- **Zero False Positives on Clean Audio:** `BandCheck` successfully differentiates naturally bandwidth-limited instruments (e.g., double bass rolling off above 4 kHz) from artificial brickwall lossy codec cutoffs.
- **Accurate Cutoff Estimation:** Cutoff boundaries are localized with an average error of only $37.7\text{ Hz}$ across the full 8–20 kHz spectrum.
- **100% Split Leakage Protection:** `SplitSafe` identified all cross-partition duplicates, preventing contaminated training sets from artificially inflating test performance.

---

## 4. Quickstart

### Installation
Install individual tools from PyPI:
```bash
pip install bandcheck phasecheck splitsafe stemcheck wavewash
```

### Command-Line Usage
```bash
# Audit an audio directory for fake upsampling and lossy cutoffs
bandcheck audit ./dataset/

# Verify train and test folders for data leakage
splitsafe check ./dataset/train ./dataset/test --manifest manifest.csv

# Audit stereo files for phase cancellation and mono compatibility
phasecheck audit ./audio_stems/
```

### Programmatic Python API
```python
from pathlib import Path
import bandcheck.check
import phasecheck.check

# 1. Spectral forensics
verdict = bandcheck.check.check_file(Path("speech_sample.wav"))
if verdict.status in ["FAKE", "LIMITED"]:
    print(f"Warning: Cutoff detected at {verdict.cutoff_hz:.1f} Hz (Lossy codec artifact)")

# 2. Phase coherence check
phase_res = phasecheck.check.check_file(Path("stereo_mix.wav"))
if phase_res.status == "INVERTED":
    print("Warning: Severe polarity inversion detected. Mono downmix will cancel out.")
```

---

## 5. Reproducing the Benchmark Evaluation

Clone this repository and run the synthetic corpus evaluation:
```bash
# 1. Generate synthetic ground-truth corpus
python generators/generate_synthetic_corpus.py

# 2. Run automated benchmark evaluation
python evaluation/run_benchmark_evaluation.py
```
Detailed results and logs will be written to `results/BENCHMARK_EVALUATION_REPORT.md`.

---

## 6. Citation

If you use this suite or its components in your research, please cite the JOSS paper:

```bibtex
@article{arjunan2026audiointegrity,
  title={{ComposerMind AudioIntegrity Suite: A Modular Python Framework for Audio Dataset Quality, Spectral Forensics, and Leakage Detection}},
  author={Arjunan, Boobalan},
  journal={Journal of Open Source Software},
  year={2026},
  note={Under review}
}
```

---

## 7. License

Released under the [MIT License](LICENSE). Copyright © 2026 Boobalan Arjunan, ComposerMind Labs.

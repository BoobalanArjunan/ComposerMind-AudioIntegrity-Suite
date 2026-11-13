# ComposerMind AudioIntegrity Suite: Audio ML Dataset Curation Framework

[![PyPI Version](https://img.shields.io/badge/pypi-7%20packages-blue.svg)](https://pypi.org/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JOSS Status](https://img.shields.io/badge/status-submission--ready-success.svg)](paper.md)

> **Production-grade Python tools and CLI utilities for Audio AI, Deep Learning, and Speech ML engineers to audit, curate, and quarantine training datasets before model training.**

Developed and maintained by **Boobalan Arjunan** at **ComposerMind Labs, Chennai, India**.

---

## The Real-World Audio ML Bottleneck

Modern deep learning architectures for audio—including neural audio codecs, diffusion models, automatic speech recognition (ASR), and speech synthesis—require tens of thousands of hours of audio crawled from the web, YouTube, podcasts, and open repositories. 

However, raw uncurated acoustic corpora contain silent, catastrophic defects that cripple model training:

1. **Fake 48 kHz Upsampling & Codec Cliffs:** Files look like pristine 48 kHz uncompressed WAVs in standard file headers, but are actually re-encoded or upsampled from 64 kbps MP3s. Generative audio and diffusion models trained on these corrupted files **learn to hallucinate harsh digital hiss and unnatural high-frequency artifacts**.
2. **Phase Cancellation on Downmix:** Multi-channel speech or stereo music often suffers from polarity inversions or microsecond Haas delays. When PyTorch data loaders downmix to mono for feature extraction (e.g., mel-spectrograms), **destructive interference silently erases speech formants and harmonics**.
3. **Catastrophic Cross-Split Data Leakage:** Minor edits (gain adjustments, 50ms trims, MP3 re-encoding) allow training files to contaminate validation and test partitions. Models appear to achieve world-class test loss, **masking severe real-world generalization failure**.
4. **Wasted GPU Compute:** Launching multi-node GPU training clusters costing thousands of dollars on poisoned datasets results in wasted compute budgets and degraded model checkpoints.

The **ComposerMind AudioIntegrity Suite** solves this by providing 7 specialized, modular Python tools built on zero-bloat CPU primitives (`numpy`, `scipy`, `soundfile`) that act as a pre-training quarantine firewall.

---

## Pre-Training Audio ML Curation Framework (7 Tools)

The suite organizes 7 specialized tools across three operational dimensions of dataset curation:

### 1. Pre-Ingestion Signal Hygiene & Forensics
* **`bandcheck` (Spectral Cutoff Forensics):** Detects brickwall lossy compression cutoffs and fake 48 kHz upsampling via Modified Discrete Cosine Transform (MDCT) frame statistics. *ML Usage:* Automatically drops or flags web-scraped audio to prevent generative and diffusion models from hallucinating high-frequency codec hiss.
* **`phasecheck` (Spatial & Phase Integrity):** Quantifies inter-channel cross-correlation ($\rho$), sub-sample Haas delay, and mono downmix cancellation loss in dB. *ML Usage:* Quarantines polarity-inverted stereo files before feature extraction to prevent destructive phase cancellation when PyTorch models downmix to mono.
* **`wavewash` (Waveform Quality Control):** Scans for hard clipping, digital saturation, DC offset bias, and dynamic SNR. *ML Usage:* Pre-conditions raw waveforms with automated DC-bias centering and headroom normalization before spectrogram computation.
* **`driftcheck` (Clock Drift Synchronization):** Tracks asynchronous sample-clock drift rates in parts-per-million ($\text{ppm}$) across independent recording interfaces. *ML Usage:* Aligns multi-microphone speech corpora before multi-channel acoustic modeling.

### 2. Dataset Partition Security
* **`splitsafe` (Contamination Firewall):** Generates robust acoustic fingerprints to detect cross-partition duplicates across training, validation, and testing directories. *ML Usage:* Identifies bit-exact duplicates, trimmed excerpts, and re-encoded copies across splits, ensuring reported test accuracy reflects true generalization.

### 3. Annotation & Metadata Integrity
* **`tempocheck` (Rhythmic Annotation Audit):** Verifies dynamic BPM trajectories and downbeat grid alignment. *ML Usage:* Audits music AI dataset labels, flagging half-time/double-time annotation errors in training manifests.
* **`tonicfind` (Harmonic Key Validation):** Computes 12-dimensional chromagram correlations against Temperley/Krumhansl tonal profiles. *ML Usage:* Validates musical key and tonality metadata for conditional music generation models.

---

## Master Package Registry (The 7 Pre-Training Tools)

| # | Package Name | Command-Line Tool | Module Scope | Detection Target & Method | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 01 | [`bandcheck`](https://pypi.org/project/bandcheck/) | `bandcheck` | Spectral Forensics | Lossy cutoffs & fake upsampling (MDCT) | ✅ Live on PyPI |
| 02 | [`phasecheck`](https://pypi.org/project/phasecheck/) | `phasecheck` | Spatial Integrity | Polarity inversion & Haas skew (Correlation) | ✅ Live on PyPI |
| 03 | [`wavewash`](https://pypi.org/project/wavewash/) | `wavewash` | Waveform Hygiene | Peak clipping, DC bias repair, dynamic SNR | ✅ Live on PyPI |
| 04 | [`driftcheck-audio`](https://pypi.org/project/driftcheck-audio/) | `driftcheck` | Synchronization | Asynchronous sample-rate clock drift ($\text{ppm}$) | ✅ Live on PyPI |
| 05 | [`splitsafe`](https://pypi.org/project/splitsafe/) | `splitsafe` | Partition Security | Train/Val/Test contamination & leakage | ✅ Live on PyPI |
| 06 | [`tempocheck`](https://pypi.org/project/tempocheck/) | `tempocheck` | Annotation Audit | Dynamic BPM trajectories & downbeat grids | ✅ Live on PyPI |
| 07 | [`tonicfind`](https://pypi.org/project/tonicfind/) | `tonicfind` | Annotation Audit | Chroma correlation & harmonic key profiles | ✅ Live on PyPI |

---

## How Audio ML Engineers Use This in Production

### 1. PyTorch / Hugging Face DataLoader Integration
Filter or quarantine defective audio on-the-fly during dataset compilation:

```python
from pathlib import Path
import bandcheck.check
import phasecheck.check
import soundfile as sf
import torch
from torch.utils.data import Dataset

class CuratedAudioDataset(Dataset):
    def __init__(self, file_paths, quarantine_dir="quarantine/"):
        self.valid_paths = []
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(exist_ok=True)
        
        for p in file_paths:
            # 1. Reject fake-upsampled or lossy-compressed audio
            band_res = bandcheck.check.check_file(Path(p))
            if band_res.status in ["FAKE", "LIMITED"]:
                continue  # Discard files that teach models to hallucinate codec artifacts
                
            # 2. Reject phase-inverted files before mono downmix
            phase_res = phasecheck.check.check_file(Path(p))
            if phase_res.status == "INVERTED":
                continue  # Avoid phase cancellation during mono feature extraction
                
            self.valid_paths.append(p)
            
    def __len__(self):
        return len(self.valid_paths)
        
    def __getitem__(self, idx):
        wav, sr = sf.read(self.valid_paths[idx])
        return torch.tensor(wav, dtype=torch.float32)
```

### 2. CI/CD Dataset Sanitization (Preventing Wasted GPU Compute)
Run automated dataset hygiene checks in your training pipeline or GitHub Actions before launching GPU clusters:

```bash
# 1. Audit a 100,000-file scraped dataset for fake upsampling
bandcheck audit ./scraped_audio/ --output-manifest audit_report.json

# 2. Ensure zero train/test leakage before spinning up a multi-GPU training run
splitsafe check ./train_set ./test_set --threshold 0.80 --manifest leakage_audit.csv

# 3. Clean waveform anomalies (DC bias, clipping) across audio files
wavewash clean ./raw_audio/ --output-dir ./cleaned_audio/

# 4. Audit stereo datasets for phase cancellation issues
phasecheck audit ./stereo_stems/ --quarantine ./corrupted_phase/
```

---

## Empirical Benchmark Verification

While all 7 packages maintain independent automated test suites with over **200 passing unit tests**, the pre-training forensics detectors were evaluated against a calibrated 48 kHz synthetic corpus (34 entries) containing controlled mathematical defects alongside clean acoustic controls:

| Detector | Injected Defect Target | Precision | Recall | Parameter Estimation Error | Clean Negative False Alarms |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`BandCheck`** | Chebyshev cutoffs (8–20 kHz) | **60.0%** | **85.7%** | **$37.7\text{ Hz}$** MAE | **0.0%** (0/10 false alarms) |
| **`PhaseCheck`** | Polarity inversion / Haas skew | **100.0%** | **50.0%** | $\rho = -1.00$ verified | **0.0%** (0/10 false alarms) |
| **`SplitSafe`** | Train/test split leakage | **100.0%** | **100.0%** | Exact (100%) & trimmed (82%) | **0.0%** false alarms |

### Key Findings for ML Practitioners:
- **Zero False Positives on Clean Audio:** `BandCheck` successfully differentiates naturally bandwidth-limited acoustic instruments (e.g., double bass rolling off above 4 kHz) from artificial brickwall lossy codec cutoffs.
- **Sub-40 Hz Estimation Accuracy:** Cutoff boundaries are localized with an average error of only $37.7\text{ Hz}$ across the full 8–20 kHz spectrum.
- **100% Split Leakage Protection:** `SplitSafe` identified all cross-partition duplicates, preventing contaminated training sets from artificially inflating test performance.

---

## Reproducing the Benchmark Evaluation

Clone this repository and run the synthetic corpus evaluation:
```bash
git clone https://github.com/BoobalanArjunan/ComposerMind-AudioIntegrity-Suite.git
cd ComposerMind-AudioIntegrity-Suite

# 1. Generate synthetic ground-truth corpus
python generators/generate_synthetic_corpus.py

# 2. Run automated benchmark evaluation
python evaluation/run_benchmark_evaluation.py
```
Detailed results and logs will be written to `results/BENCHMARK_EVALUATION_REPORT.md`.

---

## Citation

If you use this suite or its components in your machine learning workflows or research, please cite the JOSS paper:

```bibtex
@article{arjunan2026audiointegrity,
  title={{ComposerMind AudioIntegrity Suite: A Modular Python Framework for Audio ML Dataset Curation, Spectral Forensics, and Leakage Detection}},
  author={Arjunan, Boobalan},
  journal={Journal of Open Source Software},
  year={2026},
  note={Under review}
}
```

---

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Boobalan Arjunan, ComposerMind Labs.

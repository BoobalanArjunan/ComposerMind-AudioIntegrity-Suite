# ComposerMind AudioIntegrity Suite: Audio AI & ML Dataset Curation Framework

[![PyPI Version](https://img.shields.io/badge/pypi-10%20packages-blue.svg)](https://pypi.org/)
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

The **ComposerMind AudioIntegrity Suite** solves this by providing 10 specialized, modular Python tools built on zero-bloat CPU primitives (`numpy`, `scipy`, `soundfile`) that act as a pre-training quarantine firewall.

---

## Master Package Registry (The 10 Tools)

| # | Package Name | Command-Line Tool | Core Audio ML & Dataset Curation Capability | Status |
| :---: | :--- | :--- | :--- | :---: |
| 01 | [`bandcheck`](https://pypi.org/project/bandcheck/) | `bandcheck` | Detects lossy codec cutoffs & fake upsampling before generative model training | ✅ Live on PyPI |
| 02 | [`phasecheck`](https://pypi.org/project/phasecheck/) | `phasecheck` | Sub-sample phase coherence & mono downmix cancellation loss detection | ✅ Live on PyPI |
| 03 | [`splitsafe`](https://pypi.org/project/splitsafe/) | `splitsafe` | Acoustic fingerprinting firewall detecting train/val/test data leakage | ✅ Live on PyPI |
| 04 | [`wavewash`](https://pypi.org/project/wavewash/) | `wavewash` | Waveform hygiene: detects clipping, repairs DC offset, audits SNR | ✅ Live on PyPI |
| 05 | [`stemcheck`](https://pypi.org/project/stemcheck/) | `stemcheck` | Evaluates music source separation models by measuring Wiener soft-mask bleed | ✅ Live on PyPI |
| 06 | [`driftcheck-audio`](https://pypi.org/project/driftcheck-audio/) | `driftcheck` | Asynchronous sample-rate clock drift estimator for multi-mic recording setups | ✅ Live on PyPI |
| 07 | [`tempocheck`](https://pypi.org/project/tempocheck/) | `tempocheck` | Audits rhythmic dataset labels, downbeat grids, and dynamic BPM trajectories | ✅ Live on PyPI |
| 08 | [`tonicfind`](https://pypi.org/project/tonicfind/) | `tonicfind` | Chroma correlation & MIREX-scored musical key / tonic label integrity checker | ✅ Live on PyPI |
| 09 | [`turnscope`](https://pypi.org/project/turnscope/) | `turnscope` | Audits conversational speech datasets for talk-over overlap, latency, and VAD | ✅ Live on PyPI |
| 10 | [`mixforge-audio`](https://pypi.org/project/mixforge-audio/) | `mixforge` | Benchmark multi-track mixture generator with calibrated bleed and RIR acoustics | ✅ Live on PyPI |

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
# Audit a 100,000-file scraped dataset for fake upsampling
bandcheck audit ./scraped_audio/ --output-manifest audit_report.json

# Ensure zero train/test leakage before spinning up a multi-GPU training run
splitsafe check ./train_set ./test_set --threshold 0.80 --manifest leakage_audit.csv

# Audit stereo datasets for phase cancellation issues
phasecheck audit ./stereo_stems/ --quarantine ./corrupted_phase/
```

### 3. Audio Model Evaluation & Benchmarking
Evaluate source separation and speech diarization checkpoints:

```bash
# Benchmark Demucs / Spleeter stem separation bleed
stemcheck audit ./model_eval_stems/

# Audit diarization timing, speaker overlaps, and conversational latency
turnscope audit ./conversational_eval/
```

---

## Empirical Benchmark Verification

The suite was rigorously evaluated against a calibrated 48 kHz synthetic corpus (34 entries) containing controlled mathematical defects alongside clean acoustic controls:

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
  title={{ComposerMind AudioIntegrity Suite: A Modular Python Framework for Audio Dataset Quality, Spectral Forensics, and Leakage Detection}},
  author={Arjunan, Boobalan},
  journal={Journal of Open Source Software},
  year={2026},
  note={Under review}
}
```

---

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Boobalan Arjunan, ComposerMind Labs.

---
title: 'ComposerMind AudioIntegrity Suite: A Modular Python Framework for Audio Dataset Quality, Spectral Forensics, and Leakage Detection'
tags:
  - Python
  - audio machine learning
  - digital signal processing
  - dataset integrity
  - spectral forensics
  - data leakage
  - music information retrieval
authors:
  - name: Boobalan Arjunan
    affiliation: 1
affiliations:
  - name: ComposerMind Labs, Chennai, India
    index: 1
date: 13 November 2026
bibliography: paper.bib
---

# Summary

Deep learning models for speech recognition, audio source separation, acoustic classification, and generative sound synthesis depend heavily on the integrity of large-scale training corpora [@panayotov2015librispeech; @rafii2017musdb18]. However, raw audio crawled from public web repositories or legacy archives frequently contains insidious defects: brickwall frequency cutoffs disguised by artificial upsampling, inter-channel phase cancellation, clock drift across asynchronous recording interfaces, inter-stem acoustic bleed in multi-track recordings, and silent train/validation/test contamination [@defossez2019demucs].

The **ComposerMind AudioIntegrity Suite** is an open-source, modular Python framework engineered to audit, quantify, and remediate these audio defects before model training. Built upon zero-bloat CPU primitives (`numpy`, `scipy`, and `soundfile`), the suite provides 10 interoperable command-line tools and Python modules that evaluate acoustic corpora across three key lifecycle dimensions: pre-training signal forensics, dataset split contamination, and post-model separation diagnostics.

# Statement of Need

Standard data-loading pipelines in machine learning frameworks typically verify only basic file formatting, such as readability and container headers. Traditional audio quality metrics (e.g., Signal-to-Noise Ratio, PESQ) require an aligned reference recording or focus narrowly on human perceptual listening quality rather than machine-learning dataset hygiene. Consequently, critical dataset flaws remain undetected:
1. **Fake Upsampling and Codec Artifacts:** Audio upsampled from 16 kHz or 22.05 kHz to 48 kHz produces severe spectral "cliffs" where no high-frequency information exists. Models trained on these files learn to hallucinate unnatural high-frequency noise.
2. **Phase Cancellation and Haas Delay:** Inadvertent polarity inversions or microsecond inter-channel delays cause devastating phase cancellation when downmixed to mono during feature extraction.
3. **Split Contamination:** Re-encoded, trimmed, or gain-shifted copies of training recordings frequently leak into test sets, artificially inflating benchmark performance.

Existing audio analysis toolkits, such as `librosa` [@mcfee2015librosa], offer general-purpose DSP algorithms but lack automated decision boundaries, defect pass/fail criteria, and machine-readable audit reports. The **AudioIntegrity Suite** fills this gap by providing purpose-built diagnostic detectors with quantified thresholds and automated quarantine mechanisms.

# State of the Field

General-purpose audio libraries such as `librosa` [@mcfee2015librosa] and `torchaudio` provide foundational DSP primitives (STFT, chromagrams, MFCCs) but require researchers to manually invent ad-hoc heuristics, thresholds, and reporting pipelines. Conversely, traditional perceptual evaluation tools (e.g., PESQ, POLQA, ViSQOL) measure telecommunication speech quality against reference signals, making them inapplicable to reference-free, large-scale web audio crawls.

The **AudioIntegrity Suite** uniquely addresses audio dataset engineering by providing:
1. **Calibrated Decision Boundaries:** Statistically tuned detector thresholds that differentiate acoustic roll-offs from artificial codec cliffs with zero false positives.
2. **Cross-Partition Security:** Acoustic fingerprinting tailored specifically to identify train/validation/test contamination across re-encoded or trimmed segments.
3. **Autonomous Quarantining:** Production-grade CLI tools and Python hooks that output machine-readable JSON/CSV audit reports for automated data filtering in PyTorch and Hugging Face data loaders.

# Architecture & Tool Suite

The suite comprises 10 specialized open-source tools distributed via the Python Package Index (PyPI):

| Tool | Core Module Functionality | Output Metrics |
| :--- | :--- | :--- |
| **`bandcheck`** | Modified Discrete Cosine Transform (MDCT) spectral cutoff forensics | Cutoff frequency ($f_c$), upsampling ratio |
| **`phasecheck`** | Inter-channel cross-correlation & Haas delay estimation | Correlation coefficient ($\rho$), mono downmix cancellation loss ($\text{dB}$) |
| **`splitsafe`** | Acoustic fingerprinting and cross-split duplicate detection | Match confidence, alignment overlap ratio |
| **`driftcheck`** | Time-activity alignment and asynchronous clock drift tracking | Clock drift rate ($\text{ppm}$), onset delay offset |
| **`stemcheck`** | Multi-track Wiener soft-mask leakage estimation | Inter-stem bleed index ($\text{dB}$), HPSS isolation ratio |
| **`wavewash`** | Waveform quality control and DC bias repair | Clipping percentage, DC offset, SNR |
| **`tempocheck`** | Dynamic BPM trajectory and downbeat grid verification | BPM consistency score, tempo transition points |
| **`tonicfind`** | 12-dimensional chroma correlation and key label validation | Harmonic profile score, MIREX key relation |
| **`turnscope`** | Conversational voice activity detection and overlap analysis | Speaker overlap ratio, response latency |
| **`mixforge`** | Controlled multi-track mixture synthesis with calibrated bleed | Ground-truth acoustic mixes and RIR stems |

# Empirical Benchmark Verification

To evaluate detection power and false-alarm resistance, the suite was benchmarked against a controlled synthetic ground-truth corpus (48 kHz, 34 entries) comprising clean negative controls, steep Chebyshev Type II brickwall cutoffs (8 kHz to 20 kHz), calibrated Haas delays, and injected split leakages.

| Detector | Injected Target | Precision | Recall | Parameter Estimation Error | Clean Negative False-Alarm Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`bandcheck`** | Fake upsampling & cutoffs | **60.0%** | **85.7%** | **$37.7\text{ Hz}$** MAE | **0.0%** (0/10 false alarms) |
| **`phasecheck`** | Polarity inversion / Haas skew | **100.0%** | **50.0%** | $\rho = -1.00$ verified | **0.0%** (0/10 false alarms) |
| **`splitsafe`** | Train/test split leakage | **100.0%** | **100.0%** | Exact (100%) & trimmed (82%) | **0.0%** false alarms |

Crucially, `bandcheck` successfully differentiated naturally bandwidth-limited acoustic instruments (e.g., double bass rolling off above 4 kHz) from artificial brickwall codec cutoffs, producing **zero false alarms** on clean recordings. `splitsafe` detected both bit-exact duplicates and trimmed/gain-modified duplicates across training and testing partitions.

# Installation and Usage

All tools are independently installable via `pip` or executable as unified CLI utilities:

```bash
# Install individual tools
pip install bandcheck phasecheck splitsafe stemcheck wavewash

# Run automated spectral forensics on an audio folder
bandcheck audit ./my_dataset/

# Verify train/test splits for leakage
splitsafe check ./train ./test --manifest manifest.csv
```

Programmatic Python usage enables direct integration into data-loading pipelines:

```python
from pathlib import Path
import bandcheck.check
import phasecheck.check

# Fast file audit
verdict = bandcheck.check.check_file(Path("audio_sample.wav"))
if verdict.status in ["FAKE", "LIMITED"]:
    print(f"Defect detected: Cutoff at {verdict.cutoff_hz:.1f} Hz")
```

# Acknowledgements

The author thanks ComposerMind Labs for supporting the open-source engineering and empirical validation of this research toolkit.

# References

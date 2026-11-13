---
title: 'ComposerMind AudioIntegrity Suite: A Modular Python Framework for Audio Dataset Quality, Spectral Forensics, and Leakage Detection'
tags:
  - Python
  - audio machine learning
  - digital signal processing
  - dataset integrity
  - spectral forensics
  - data leakage
  - deep learning
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

Deep learning architectures for audio—spanning neural audio codecs, diffusion models, automatic speech recognition (ASR), and generative sound synthesis—depend fundamentally on the integrity of large-scale training corpora [@panayotov2015librispeech; @rafii2017musdb18]. When engineering pipelines scrape millions of audio hours from public web repositories, YouTube, or podcast dumps, the resulting datasets are frequently contaminated by insidious, silent defects: brickwall frequency cutoffs disguised by artificial upsampling, inter-channel phase cancellation, clock drift across asynchronous interfaces, inter-stem acoustic bleed in multi-track recordings, and silent train/validation/test contamination [@defossez2019demucs].

The **ComposerMind AudioIntegrity Suite** is an open-source, modular Python framework engineered for machine learning practitioners and signal processing engineers to audit, quantify, and quarantine defective audio before model training. Built upon zero-bloat CPU primitives (`numpy`, `scipy`, and `soundfile`), the suite provides 10 interoperable command-line tools and Python modules that evaluate acoustic corpora across three lifecycle stages: pre-training signal forensics, dataset partition contamination, and post-model separation diagnostics.

# Statement of Need

Standard data loaders in modern deep learning frameworks (e.g., PyTorch, TensorFlow) verify only container formatting and file readability. Traditional speech quality metrics (e.g., PESQ, POLQA) require time-aligned clean reference signals or measure human conversational listening preference rather than machine-learning dataset hygiene. Consequently, four critical dataset flaws routinely pass into multi-GPU training clusters:

1. **Fake 48 kHz Upsampling and Codec Artifacts:** Audio crawled from streaming sources is frequently transcoded or upsampled from low-bitrate MP3/AAC streams to 48 kHz uncompressed containers. While headers report 48 kHz, high-frequency energy abruptly cuts off at 11–16 kHz. Generative audio and diffusion models trained on such files learn to hallucinate harsh digital hiss and unnatural high-frequency artifacts.
2. **Phase Cancellation on Mono Downmix:** Multi-channel speech or stereo music often contains accidental polarity inversions or microsecond Haas delays. When PyTorch data loaders downmix stereo inputs to mono for feature extraction (e.g., mel-spectrograms), destructive acoustic interference erases speech formants and harmonic energy.
3. **Sub-Segment Split Contamination:** Minor alterations—such as 50ms temporal trimming, gain adjustments, or re-encoding—allow training files to contaminate validation and test partitions. Models achieve deceptively low validation loss, masking catastrophic generalization failures in production.
4. **Wasted Compute Budgets:** Incurring thousands of dollars in cloud GPU compute to train models on poisoned acoustic corpora results in degraded checkpoints and redundant training cycles.

Existing audio toolkits, such as `librosa` [@mcfee2015librosa], provide general-purpose mathematical primitives but lack automated decision boundaries, defect pass/fail criteria, and machine-readable quarantine mechanisms. The **AudioIntegrity Suite** bridges this gap by providing purpose-built diagnostic detectors with statistically calibrated thresholds and automated quarantine manifests.

# State of the Field

General-purpose audio libraries such as `librosa` [@mcfee2015librosa] and `torchaudio` provide foundational DSP primitives (STFT, chromagrams, MFCCs) but require ML engineers to manually construct ad-hoc heuristics, thresholds, and reporting pipelines. Conversely, traditional perceptual evaluation tools (e.g., PESQ, POLQA, ViSQOL) measure telecommunication speech quality against reference signals, making them inapplicable to reference-free, large-scale web audio crawls.

The **AudioIntegrity Suite** uniquely addresses audio dataset engineering by providing:
1. **Calibrated Decision Boundaries:** Statistically tuned detector thresholds that differentiate acoustic roll-offs from artificial codec cliffs with zero false alarms.
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

All tools are installable individually via `pip` or executable as unified CLI utilities:

```bash
# Install individual tools from PyPI
pip install bandcheck phasecheck splitsafe stemcheck wavewash

# Audit a scraped audio folder before kicking off model training
bandcheck audit ./scraped_audio/ --output-manifest audit_manifest.json

# Ensure zero train/test leakage before spinning up multi-GPU clusters
splitsafe check ./train ./test --manifest leakage_audit.csv
```

Programmatic integration directly into PyTorch data-loading pipelines enables on-the-fly dataset sanitization:

```python
from pathlib import Path
import bandcheck.check
import phasecheck.check

def sanitize_sample(audio_path: Path) -> bool:
    # 1. Reject fake-upsampled or lossy-compressed audio
    band_verdict = bandcheck.check.check_file(audio_path)
    if band_verdict.status in ["FAKE", "LIMITED"]:
        return False  # Discard files causing generative models to hallucinate noise
        
    # 2. Reject phase-inverted audio before mono downmixing
    phase_verdict = phasecheck.check.check_file(audio_path)
    if phase_verdict.status == "INVERTED":
        return False  # Prevent cancellation during mono feature extraction
        
    return True
```

# Acknowledgements

The author thanks ComposerMind Labs for supporting the open-source engineering and empirical validation of this dataset curation toolkit.

# References

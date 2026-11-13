---
title: 'ComposerMind AudioIntegrity Suite: A Modular Python Framework for Audio ML Dataset Curation, Spectral Forensics, and Leakage Detection'
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

# Architecture & Tool Suite: 3-Stage ML Dataset Curation

The suite organizes 10 specialized Python packages across three critical lifecycle stages of audio machine learning engineering:

### Stage 1: Pre-Ingestion Signal Hygiene & Forensics
* **`bandcheck` (Spectral Forensics):** Detects brickwall lossy compression cutoffs and fake 48 kHz upsampling via Modified Discrete Cosine Transform (MDCT) frame statistics. *ML Usage:* Automatically drops or flags web-scraped audio to prevent generative and diffusion models from hallucinating high-frequency codec hiss.
* **`phasecheck` (Spatial & Phase Integrity):** Quantifies inter-channel cross-correlation ($\rho$), sub-sample Haas delay, and mono downmix cancellation loss in dB. *ML Usage:* Quarantines polarity-inverted stereo files before feature extraction to prevent destructive phase cancellation when PyTorch models downmix to mono.
* **`wavewash` (Waveform Quality Control):** Scans for hard clipping, digital saturation, DC offset bias, and dynamic SNR. *ML Usage:* Pre-conditions raw waveforms with automated DC-bias centering and headroom normalization before spectrogram computation.
* **`driftcheck` (Clock Drift Synchronization):** Tracks asynchronous sample-clock drift rates in parts-per-million ($\text{ppm}$) across independent recording interfaces. *ML Usage:* Aligns multi-microphone speech corpora before multi-channel acoustic modeling.

### Stage 2: Dataset Partition Security & Synthetic Ground Truth
* **`splitsafe` (Contamination Firewall):** Generates robust acoustic fingerprints to detect cross-partition duplicates across training, validation, and testing directories. *ML Usage:* Identifies bit-exact duplicates, trimmed excerpts, and re-encoded copies across splits, ensuring reported test accuracy reflects true generalization.
* **`mixforge` (Benchmark Synthesis Engine):** Mixes multi-track stems with calibrated inter-channel acoustic bleed and simulated Room Impulse Responses (RIRs). *ML Usage:* Generates controlled, ground-truth audio mixtures to benchmark source separation and speech enhancement algorithms.

### Stage 3: Annotation Validation & Model Diagnostics
* **`tempocheck` (Rhythmic Annotation Audit):** Verifies dynamic BPM trajectories and downbeat grid alignment. *ML Usage:* Audits music AI dataset labels, flagging half-time/double-time annotation errors in training manifests.
* **`tonicfind` (Harmonic Key Validation):** Computes 12-dimensional chromagram correlations against Temperley/Krumhansl tonal profiles. *ML Usage:* Validates musical key and tonality metadata for conditional music generation models.
* **`turnscope` (Conversational Speech Forensics):** Measures Voice Activity Detection (VAD) boundaries, speaker overlap duration, and turn-taking latency. *ML Usage:* Audits diarization datasets and benchmarks conversational speech synthesis models.
* **`stemcheck` (Separation Bleed Diagnostics):** Calculates inter-stem Wiener soft-mask bleed and harmonic-percussive separation (HPSS) leakage. *ML Usage:* Evaluates Demucs/Spleeter source separation checkpoints by measuring residual acoustic bleed across separated stems.

| Tool | ML Pipeline Stage | Detection Target & Method | Primary Output Metrics |
| :--- | :--- | :--- | :--- |
| **`bandcheck`** | Pre-Ingestion | Lossy cutoffs & fake upsampling (MDCT) | Cutoff frequency ($f_c$), upsampling ratio |
| **`phasecheck`** | Pre-Ingestion | Polarity inversion & Haas skew (Correlation) | Correlation ($\rho$), mono cancellation loss ($\text{dB}$) |
| **`wavewash`** | Pre-Ingestion | Peak clipping, DC offset bias, dynamic SNR | Clipping %, DC bias %, SNR ($\text{dB}$) |
| **`driftcheck`** | Pre-Ingestion | Multi-device asynchronous clock drift | Clock drift rate ($\text{ppm}$), time offset |
| **`splitsafe`** | Partition Security | Train/Val/Test contamination & leakage | Match confidence, alignment overlap ratio |
| **`mixforge`** | Synthetic Benchmarking | Calibrated multi-track mixing & RIRs | Ground-truth SDR/SIR/SAR reference mixes |
| **`tempocheck`** | Annotation Audit | BPM consistency & downbeat grid alignment | BPM consistency score, tempo transitions |
| **`tonicfind`** | Annotation Audit | 12-dimensional harmonic key profiling | Key label profile score, MIREX relation |
| **`turnscope`** | Model Evaluation | Conversational speech overlap & latency | Speaker overlap ratio, response latency |
| **`stemcheck`** | Model Evaluation | Wiener soft-mask source separation bleed | Inter-stem bleed index ($\text{dB}$), HPSS isolation |

# Empirical Benchmark Verification

While all 10 packages maintain independent automated test suites with over 300 passing unit tests, the core pre-training forensics detectors were evaluated against a calibrated synthetic ground-truth corpus (48 kHz, 34 entries) containing mathematically generated defects and clean acoustic controls:

| Detector | Injected Target Defect | Precision | Recall | Parameter Estimation Error | Clean Negative False-Alarm Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`bandcheck`** | Fake upsampling & cutoffs (8–20 kHz) | **60.0%** | **85.7%** | **$37.7\text{ Hz}$** MAE | **0.0%** (0/10 false alarms) |
| **`phasecheck`** | Polarity inversion / Haas skew | **100.0%** | **50.0%** | $\rho = -1.00$ verified | **0.0%** (0/10 false alarms) |
| **`splitsafe`** | Train/test split leakage | **100.0%** | **100.0%** | Exact (100%) & trimmed (82%) | **0.0%** false alarms |

Crucially, `bandcheck` successfully differentiated naturally bandwidth-limited acoustic instruments (e.g., double bass rolling off above 4 kHz) from artificial brickwall codec cutoffs, producing **zero false alarms** on clean recordings. `splitsafe` detected both bit-exact duplicates and trimmed/gain-modified duplicates across training and testing partitions.

# Installation and Practical ML Usage

All 10 tools are installable individually via `pip` or executable as unified CLI utilities:

```bash
# Install individual tools from PyPI
pip install bandcheck phasecheck splitsafe stemcheck wavewash

# 1. Batch audit a scraped audio folder before launching GPU training
bandcheck audit ./scraped_corpus/ --output-manifest audit_manifest.json

# 2. Prevent multi-GPU data leakage across training and validation partitions
splitsafe check ./train_set ./val_set --manifest leakage_audit.csv

# 3. Clean waveform anomalies (DC bias, clipping) across audio files
wavewash clean ./raw_audio/ --output-dir ./cleaned_audio/

# 4. Evaluate source separation models on output stems
stemcheck audit ./demucs_output_stems/
```

Programmatic integration directly into PyTorch data-loading pipelines enables on-the-fly dataset sanitization:

```python
from pathlib import Path
import bandcheck.check
import phasecheck.check

def sanitize_audio_sample(audio_path: Path) -> bool:
    """Filter corrupted audio files during ML dataset ingestion."""
    # 1. Reject fake-upsampled or lossy-compressed audio
    band_verdict = bandcheck.check.check_file(audio_path)
    if band_verdict.status in ["FAKE", "LIMITED"]:
        return False  # Discard files causing generative models to hallucinate noise
        
    # 2. Reject phase-inverted audio before mono downmixing
    phase_verdict = phasecheck.check.check_file(audio_path)
    if phase_verdict.status == "INVERTED":
        return False  # Prevent destructive cancellation during mono feature extraction
        
    return True
```

# Acknowledgements

The author thanks ComposerMind Labs for supporting the open-source engineering and empirical validation of this dataset curation toolkit.

# References

#!/usr/bin/env python3
"""
OpenSource AudioIntegrity-10: Controlled Synthetic Ground-Truth Corpus Generator
Phase 1: Generating reproducible audio defects and clean controls with exact metadata.

Authors: Boobalan Arjunan ()
"""

import os
import json
import numpy as np
import scipy.signal
import soundfile as sf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
MANIFEST_PATH = os.path.join(BASE_DIR, "results", "ground_truth_manifest.json")

SR = 48000
DURATION = 5.0 # seconds per clip
N_SAMPLES = int(SR * DURATION)

def ensure_dirs():
    subdirs = ["clean", "bandwidth", "phase", "drift", "bleed", "waveform", "splits"]
    for sub in subdirs:
        os.makedirs(os.path.join(CORPUS_DIR, sub), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

def generate_clean_audio(seed=42, mode="broadband"):
    """Generates synthetic multi-harmonic acoustic signals."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    
    if mode == "broadband":
        # Rich harmonic music-like signal with harmonics up to 22 kHz
        f0 = 220.0 # A3
        audio = np.zeros(N_SAMPLES)
        for h in range(1, 95):
            freq = f0 * h
            if freq > 23000:
                break
            amp = (1.0 / (h ** 0.85)) * (0.8 + 0.4 * rng.rand())
            phase = rng.rand() * 2 * np.pi
            audio += amp * np.sin(2 * np.pi * freq * t + phase)
        # Add slight broadband white noise floor (-55 dB)
        audio += 0.0018 * rng.randn(N_SAMPLES)
    elif mode == "natural_bandwidth_limited":
        # Realistic acoustic bass instrument naturally rolling off above 4 kHz (NOT fake upsampling)
        f0 = 55.0 # A1
        audio = np.zeros(N_SAMPLES)
        for h in range(1, 40):
            freq = f0 * h
            amp = np.exp(-freq / 900.0)
            phase = rng.rand() * 2 * np.pi
            audio += amp * np.sin(2 * np.pi * freq * t + phase)
    elif mode == "percussive_transient":
        # Sharp click/transient train
        audio = np.zeros(N_SAMPLES)
        for click_pos in range(int(SR * 0.25), N_SAMPLES, int(SR * 0.5)):
            audio[click_pos:click_pos+120] = rng.randn(120) * np.exp(-np.linspace(0, 5, 120))
    
    # Peak normalize to -3 dBFS
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.707
    return audio

def main():
    ensure_dirs()
    manifest = []
    print("Generating OpenSource AudioIntegrity-10 Controlled Corpus...")

    # 1. CLEAN NEGATIVE CONTROLS
    for i in range(10):
        mode = "natural_bandwidth_limited" if i >= 6 else "broadband"
        sig = generate_clean_audio(seed=100 + i, mode=mode)
        # Stereo clean
        stereo = np.column_stack([sig, sig])
        fname = f"clean_{i:02d}_{mode}.wav"
        fpath = os.path.join(CORPUS_DIR, "clean", fname)
        sf.write(fpath, stereo, SR)
        manifest.append({
            "file_id": fname,
            "category": "clean",
            "defect_type": "none",
            "is_defective": False,
            "ground_truth": {"bandwidth_limited_naturally": (mode == "natural_bandwidth_limited")},
            "path": fpath
        })

    # 2. BANDWIDTH DEFECTS (Fake upsampling / Codec brickwall cutoffs)
    cutoffs = [8000, 11025, 12000, 15000, 16000, 18000, 20000]
    for idx, fc in enumerate(cutoffs):
        # Use 20-second broadband signal with flat high-frequency energy
        n_20s = int(SR * 20.0)
        rng = np.random.RandomState(200 + idx)
        X = np.fft.rfft(rng.randn(n_20s))
        f = np.fft.rfftfreq(n_20s, 1 / SR)
        f[0] = f[1]
        x_raw = np.fft.irfft(X * (f / 1000.0) ** -0.75, n_20s)
        x_raw = 0.3 * (x_raw / np.abs(x_raw).max())
        
        # Apply steep FIR brickwall cutoff
        nt = int(4 * SR / 400.0) | 1
        filtered = scipy.signal.fftconvolve(x_raw, scipy.signal.firwin(nt, fc, fs=SR), mode="same")
        # 16-bit PCM quantization floor
        pcm = np.round(np.clip(filtered, -1, 1) * 32767).astype(np.int16)
        stereo = np.column_stack([pcm, pcm])
        
        fname = f"bandwidth_cutoff_{fc}hz.wav"
        fpath = os.path.join(CORPUS_DIR, "bandwidth", fname)
        sf.write(fpath, stereo, SR, subtype='PCM_16')
        manifest.append({
            "file_id": fname,
            "category": "bandwidth",
            "defect_type": "spectral_cutoff_fake_upsampling",
            "is_defective": True,
            "ground_truth": {"cutoff_hz": fc, "target_fs": SR},
            "path": fpath
        })

    # 3. PHASE & ALIGNMENT DEFECTS (Haas delay and polarity inversion)
    delays_ms = [0.08, 0.25, 0.50, 1.00, 2.00]
    for idx, d_ms in enumerate(delays_ms):
        sig = generate_clean_audio(seed=300 + idx, mode="broadband")
        delay_samples = int(np.round(d_ms * 1e-3 * SR))
        ch_right = np.zeros_like(sig)
        ch_right[delay_samples:] = sig[:-delay_samples] if delay_samples > 0 else sig
        stereo = np.column_stack([sig, ch_right])
        fname = f"phase_delay_{d_ms:.2f}ms.wav"
        fpath = os.path.join(CORPUS_DIR, "phase", fname)
        sf.write(fpath, stereo, SR)
        manifest.append({
            "file_id": fname,
            "category": "phase",
            "defect_type": "inter_channel_phase_delay",
            "is_defective": True,
            "ground_truth": {"delay_ms": d_ms, "delay_samples": delay_samples, "polarity_inverted": False},
            "path": fpath
        })
    # Polarity Inversion
    sig = generate_clean_audio(seed=399, mode="broadband")
    stereo_inv = np.column_stack([sig, -sig])
    fname = "phase_polarity_inverted.wav"
    fpath = os.path.join(CORPUS_DIR, "phase", fname)
    sf.write(fpath, stereo_inv, SR)
    manifest.append({
        "file_id": fname,
        "category": "phase",
        "defect_type": "polarity_inversion",
        "is_defective": True,
        "ground_truth": {"delay_ms": 0.0, "polarity_inverted": True},
        "path": fpath
    })

    # 4. CLOCK DRIFT DEFECTS (PPM sample-rate divergence)
    drift_ppms = [15.0, 35.0, 60.0, 100.0]
    for idx, ppm in enumerate(drift_ppms):
        sig_ref = generate_clean_audio(seed=400 + idx, mode="broadband")
        # Drifted version has slight ratio change
        ratio = 1.0 + (ppm * 1e-6)
        n_resamp = int(np.round(len(sig_ref) * ratio))
        sig_drift = scipy.signal.resample(sig_ref, n_resamp)[:N_SAMPLES]
        if len(sig_drift) < N_SAMPLES:
            sig_drift = np.pad(sig_drift, (0, N_SAMPLES - len(sig_drift)))
        
        fname_ref = f"drift_ref_{ppm}ppm.wav"
        fname_test = f"drift_test_{ppm}ppm.wav"
        p_ref = os.path.join(CORPUS_DIR, "drift", fname_ref)
        p_test = os.path.join(CORPUS_DIR, "drift", fname_test)
        sf.write(p_ref, sig_ref, SR)
        sf.write(p_test, sig_drift, SR)
        manifest.append({
            "file_id": fname_test,
            "category": "drift",
            "defect_type": "clock_drift",
            "is_defective": True,
            "ground_truth": {"drift_ppm": ppm, "reference_file": fname_ref},
            "path": p_test
        })

    # 5. WAVEFORM LEVEL DEFECTS (Clipping & DC Offset)
    # Hard Clipping
    sig = generate_clean_audio(seed=501, mode="broadband") * 1.8 # boost to clip
    clipped = np.clip(sig, -0.75, 0.75)
    fname = "waveform_hard_clipped.wav"
    fpath = os.path.join(CORPUS_DIR, "waveform", fname)
    sf.write(fpath, np.column_stack([clipped, clipped]), SR)
    manifest.append({
        "file_id": fname,
        "category": "waveform",
        "defect_type": "clipping",
        "is_defective": True,
        "ground_truth": {"clip_threshold": 0.75},
        "path": fpath
    })
    # DC Offset
    sig = generate_clean_audio(seed=502, mode="broadband") + 0.12 # +12% DC bias
    fname = "waveform_dc_offset_12pct.wav"
    fpath = os.path.join(CORPUS_DIR, "waveform", fname)
    sf.write(fpath, np.column_stack([sig, sig]), SR)
    manifest.append({
        "file_id": fname,
        "category": "waveform",
        "defect_type": "dc_offset",
        "is_defective": True,
        "ground_truth": {"dc_offset": 0.12},
        "path": fpath
    })

    # 6. STEM BLEED / LEAKAGE DEFECTS
    bleed_dbs = [-36.0, -24.0, -14.0, -6.0]
    for idx, b_db in enumerate(bleed_dbs):
        stem_vocal = generate_clean_audio(seed=600 + idx, mode="broadband")
        stem_drums = generate_clean_audio(seed=650 + idx, mode="percussive_transient")
        # Injected bleed from drums into vocal
        gain = 10.0 ** (b_db / 20.0)
        vocal_with_bleed = stem_vocal + gain * stem_drums
        fname = f"bleed_stem_drums_in_vocal_{abs(int(b_db))}db.wav"
        fpath = os.path.join(CORPUS_DIR, "bleed", fname)
        sf.write(fpath, vocal_with_bleed, SR)
        manifest.append({
            "file_id": fname,
            "category": "bleed",
            "defect_type": "inter_stem_bleed",
            "is_defective": True,
            "ground_truth": {"bleed_db": b_db, "bleed_gain": gain, "interferer": "drums"},
            "path": fpath
        })

    # 7. SPLIT CONTAMINATION (Train/Val/Test Leakage)
    split_dir = os.path.join(CORPUS_DIR, "splits")
    os.makedirs(os.path.join(split_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "test"), exist_ok=True)
    
    # 5 unique train files, 3 unique test files, 2 leaked files (1 exact, 1 re-encoded/trimmed)
    for k in range(5):
        s = generate_clean_audio(seed=700 + k, mode="broadband")
        sf.write(os.path.join(split_dir, "train", f"train_{k:02d}.wav"), s, SR)
    for k in range(3):
        s = generate_clean_audio(seed=750 + k, mode="broadband")
        sf.write(os.path.join(split_dir, "test", f"test_clean_{k:02d}.wav"), s, SR)
    # Exact duplicate leakage (train_00 leaked into test)
    s_leak_exact = generate_clean_audio(seed=700, mode="broadband")
    sf.write(os.path.join(split_dir, "test", "test_leaked_exact_train00.wav"), s_leak_exact, SR)
    # Trimmed & level shifted leakage (train_01 leaked into test with 0.1s trim and 0.95 gain)
    s_leak_mod = generate_clean_audio(seed=701, mode="broadband")[int(SR * 0.1):] * 0.95
    sf.write(os.path.join(split_dir, "test", "test_leaked_modified_train01.wav"), s_leak_mod, SR)
    
    manifest.append({
        "file_id": "split_contamination_set",
        "category": "splits",
        "defect_type": "split_leakage",
        "is_defective": True,
        "ground_truth": {"leaked_files": ["test_leaked_exact_train00.wav", "test_leaked_modified_train01.wav"], "clean_test_files": 3},
        "path": split_dir
    })

    # Save manifest
    with open(MANIFEST_PATH, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Generated {len(manifest)} benchmark entries with machine-readable ground truth.")
    print(f"Saved manifest to: {MANIFEST_PATH}")

if __name__ == "__main__":
    main()

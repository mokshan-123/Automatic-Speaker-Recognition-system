"""
Real-Time Speaker Identification - Inference Script
=====================================================
Identify who is speaking from a new audio recording.

This script:
  1. Records audio from microphone OR loads a .wav file
  2. Extracts MFCC features (using librosa - pure Python, no MATLAB needed)
  3. Runs through the trained GMM-UBM model
  4. Outputs: Speaker ID + Confidence %

Usage:
    # Identify from a wav file:
    python identify_speaker.py --audio path/to/audio.wav

    # Record from microphone and identify:
    python identify_speaker.py --record --duration 3

    # Use MATLAB-exported features directly:
    python identify_speaker.py --mat path/to/features.mat
"""

import os
import sys
import argparse
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =========================================================================
#  MFCC EXTRACTION IN PYTHON (mirrors our MATLAB implementation)
# =========================================================================
def extract_mfcc_python(audio, sr, n_mfcc=13, n_fft=512, hop_length=100, 
                         n_mels=26, fmin=0, fmax=None, pre_emphasis=0.97):
    """
    Extract 39-dimensional MFCC features using librosa.
    
    This mirrors our MATLAB from-scratch implementation:
      1. Pre-emphasis
      2. Frame + Window (done internally by librosa)
      3. FFT + Power Spectrum
      4. Mel Filterbank
      5. Log
      6. DCT → 13 MFCCs
      7. Delta + Delta-Delta → total 39 features
    
    Parameters match our MATLAB settings:
      - n_mfcc=13 (same as num_coeffs in MATLAB)
      - n_fft=512 (same as nfft in MATLAB)
      - hop_length=100 (same as hop_size in MATLAB)
      - n_mels=26 (same as num_filters in MATLAB)
      - pre_emphasis=0.97 (same as alpha in MATLAB)
    """
    if not HAS_LIBROSA:
        raise ImportError(
            "librosa is required for Python-based MFCC extraction.\n"
            "Install it with: pip install librosa"
        )
    
    # Step 1: Pre-emphasis (same as our MATLAB preemphasis.m)
    audio_emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Steps 2-8: MFCC extraction
    # librosa internally does: framing, windowing, FFT, mel filterbank, log, DCT
    mfcc_coeffs = librosa.feature.mfcc(
        y=audio_emphasized,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr // 2,
        window='hamming',      # Same as our apply_window.m
        center=False,          # Same as our frame_signal.m (no padding)
    ).T  # Transpose to (num_frames x n_mfcc) — same as MATLAB output
    
    # Step 9: Delta and Delta-Delta (same as our compute_delta.m)
    # librosa.feature.delta with width=5 corresponds to our delta_N=2
    delta_coeffs = librosa.feature.delta(mfcc_coeffs.T, width=5).T
    delta2_coeffs = librosa.feature.delta(mfcc_coeffs.T, width=5, order=2).T
    
    # Concatenate: 13 MFCC + 13 Delta + 13 Delta-Delta = 39 features
    features = np.hstack([mfcc_coeffs, delta_coeffs, delta2_coeffs])
    
    return features


# =========================================================================
#  RECORDING
# =========================================================================
def record_audio(duration=3, sr=16000):
    """Record audio from microphone."""
    if not HAS_SOUNDDEVICE:
        raise ImportError(
            "sounddevice is required for microphone recording.\n"
            "Install it with: pip install sounddevice"
        )
    
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("✓ Recording complete!\n")
    
    return audio.flatten(), sr


# =========================================================================
#  SPEAKER IDENTIFICATION
# =========================================================================
def load_model(model_dir='trained_models'):
    """Load the trained GMM-UBM model."""
    model_path = os.path.join(model_dir, 'gmm_ubm_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model found at '{model_path}'")
        print("Run 'python train_gmm_ubm.py' first to train the model!")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def identify_speaker(features, model_data):
    """
    Identify speaker from MFCC features.
    
    Returns:
        result: dict with predicted speaker, confidence, all scores
    """
    ubm = model_data['ubm']
    speaker_gmms = model_data['speaker_gmms']
    scaler = model_data['scaler']
    speaker_list = model_data['speaker_list']
    
    # Scale features
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    # Score against UBM
    ubm_score = ubm.score(features_scaled)
    
    # Score against each speaker GMM
    scores = {}
    for spk_id, gmm in speaker_gmms.items():
        spk_score = gmm.score(features_scaled)
        scores[spk_id] = spk_score - ubm_score  # Log-likelihood ratio
    
    # Find best speaker
    predicted = max(scores, key=scores.get)
    
    # Compute confidence via softmax
    score_values = np.array([scores[spk] for spk in speaker_list])
    exp_scores = np.exp(score_values - score_values.max())
    confidences = exp_scores / exp_scores.sum()
    confidence_dict = {spk: conf for spk, conf in zip(speaker_list, confidences)}
    
    return {
        'predicted_speaker': predicted,
        'confidence': confidence_dict[predicted],
        'all_confidences': confidence_dict,
        'scores': scores,
        'num_frames': features.shape[0]
    }


def print_result(result):
    """Pretty-print identification result."""
    print("\n" + "=" * 50)
    print("  SPEAKER IDENTIFICATION RESULT")
    print("=" * 50)
    print(f"\n  🎯 Predicted Speaker: {result['predicted_speaker']}")
    print(f"  📊 Confidence: {result['confidence']:.1%}")
    print(f"  📏 Frames analyzed: {result['num_frames']}")
    
    print(f"\n  All speaker scores:")
    print(f"  {'-' * 40}")
    
    sorted_conf = sorted(result['all_confidences'].items(), 
                         key=lambda x: x[1], reverse=True)
    for rank, (spk, conf) in enumerate(sorted_conf, 1):
        bar = "█" * int(conf * 30)
        marker = " ◄── IDENTIFIED" if spk == result['predicted_speaker'] else ""
        print(f"  {rank}. {spk:<6} {conf:>6.1%} {bar}{marker}")
    
    print(f"\n{'=' * 50}\n")


# =========================================================================
#  MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Identify a speaker using the trained GMM-UBM model'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio', type=str, help='Path to .wav file')
    group.add_argument('--record', action='store_true', help='Record from microphone')
    group.add_argument('--mat', type=str, help='Path to MATLAB .mat file with pre-extracted features')
    
    parser.add_argument('--duration', type=float, default=3, 
                        help='Recording duration in seconds (default: 3)')
    parser.add_argument('--model-dir', type=str, default='trained_models',
                        help='Directory with trained models')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate for recording (default: 16000)')
    
    args = parser.parse_args()
    
    # Load trained model
    print("Loading trained model...")
    model_data = load_model(args.model_dir)
    print(f"✓ Model loaded ({len(model_data['speaker_list'])} speakers enrolled)\n")
    
    # Get features
    if args.mat:
        # Load pre-extracted features from MATLAB
        print(f"Loading MATLAB features from '{args.mat}'...")
        data = sio.loadmat(args.mat)
        features = data['features']
        print(f"✓ Loaded features: {features.shape}")
        
    elif args.record:
        # Record from microphone
        audio, sr = record_audio(duration=args.duration, sr=args.sr)
        print(f"Extracting MFCC features...")
        features = extract_mfcc_python(audio, sr)
        print(f"✓ Extracted features: {features.shape}")
        
    elif args.audio:
        # Load wav file
        print(f"Loading audio from '{args.audio}'...")
        audio, sr = librosa.load(args.audio, sr=None)
        print(f"✓ Audio: {len(audio)} samples ({len(audio)/sr:.2f}s @ {sr} Hz)")
        
        print(f"Extracting MFCC features...")
        features = extract_mfcc_python(audio, sr)
        print(f"✓ Extracted features: {features.shape}")
    
    # Identify speaker
    result = identify_speaker(features, model_data)
    print_result(result)
    
    return result


if __name__ == '__main__':
    main()

"""
Quick Pipeline Test
====================
Tests the full GMM-UBM pipeline using the actual .wav files in data/train and data/test.
Extracts MFCCs using librosa (Python) instead of MATLAB .mat files.

This lets you validate the entire pipeline works BEFORE running the MATLAB export.

Usage:
    python test_pipeline.py
"""

import os
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from train_gmm_ubm import GMMUBMSystem, Config, evaluate, print_confusion_info, print_data_summary

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed. Run: pip install librosa")
    exit(1)

from identify_speaker import extract_mfcc_python


def load_wav_and_extract_mfcc(data_dir, set_name):
    """Load .wav files and extract MFCC features using Python/librosa."""
    wav_dir = os.path.join(data_dir, set_name)
    wav_files = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
    
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in '{wav_dir}'")
    
    speakers_data = {}
    for wav_file in wav_files:
        speaker_id = os.path.splitext(os.path.basename(wav_file))[0]
        
        # Load audio
        audio, sr = librosa.load(wav_file, sr=None)
        
        # Extract 39-dim MFCCs (matching our MATLAB parameters)
        features = extract_mfcc_python(audio, sr)
        speakers_data[speaker_id] = features
        
        print(f"  {speaker_id}: {len(audio)} samples ({len(audio)/sr:.2f}s) → "
              f"{features.shape[0]} frames × {features.shape[1]} features")
    
    return speakers_data


def main():
    print("\n" + "=" * 60)
    print("  PIPELINE VALIDATION TEST")
    print("  (Using librosa MFCC extraction from .wav files directly)")
    print("=" * 60 + "\n")
    
    data_dir = "data"
    
    # Check if data exists
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        print("Make sure you're running from the Final/ directory.")
        exit(1)
    
    # Extract MFCCs
    print("Extracting MFCC features from training audio...")
    train_data = load_wav_and_extract_mfcc(data_dir, "train")
    
    print("\nExtracting MFCC features from test audio...")
    test_data = load_wav_and_extract_mfcc(data_dir, "test")
    
    # Summary
    print_data_summary(train_data, test_data)
    
    # Train GMM-UBM
    config = Config()
    config.UBM_N_COMPONENTS = 16  # Good default for 8 speakers
    
    system = GMMUBMSystem(config)
    system.train(train_data)
    
    # Save models
    system.save_models()
    
    # Evaluate
    results, accuracy = evaluate(system, test_data)
    print_confusion_info(results, system.speaker_list)
    
    # Test inference on a single file
    print("=" * 60)
    print("  SINGLE-FILE INFERENCE TEST")
    print("=" * 60)
    
    test_file = glob.glob(os.path.join(data_dir, "test", "*.wav"))[0]
    true_speaker = os.path.splitext(os.path.basename(test_file))[0]
    
    audio, sr = librosa.load(test_file, sr=None)
    features = extract_mfcc_python(audio, sr)
    
    from identify_speaker import identify_speaker, load_model, print_result
    model_data = load_model()
    result = identify_speaker(features, model_data)
    print(f"\nTest file: {test_file} (true speaker: {true_speaker})")
    print_result(result)
    
    print("✓ Pipeline validation complete!\n")


if __name__ == '__main__':
    main()

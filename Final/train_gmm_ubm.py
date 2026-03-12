"""
GMM-UBM Speaker Recognition System
====================================
Train a Universal Background Model (UBM) and speaker-specific GMMs
using MFCC features exported from MATLAB.

Architecture:
    1. Load MFCC features (.mat files) for all speakers
    2. Train a Universal Background Model (UBM) on ALL training data pooled together
    3. Adapt the UBM to each speaker using MAP adaptation → speaker-specific GMMs
    4. At test time: score test utterance against all speaker GMMs
    5. Output: predicted speaker + confidence

Usage:
    python train_gmm_ubm.py              # Train + Evaluate
    python train_gmm_ubm.py --evaluate   # Evaluate only (load saved models)
"""

import os
import glob
import pickle
import argparse
import numpy as np
import scipy.io as sio
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =========================================================================
#  CONFIGURATION
# =========================================================================
class Config:
    """All hyperparameters in one place."""
    
    # Paths
    EXPORTED_DIR = "exported_features"          # MATLAB export directory
    MODEL_DIR = "trained_models"                # Where to save trained models
    
    # GMM-UBM Parameters
    UBM_N_COMPONENTS = 16       # Number of Gaussian components in UBM
                                 # For 8 speakers with limited data: 16-32 is good
                                 # For larger datasets: 64, 128, 256, 512, 1024
    
    COVARIANCE_TYPE = 'diag'    # 'diag' is standard for speaker recognition
                                 # (full covariance needs too much data)
    
    UBM_MAX_ITER = 200          # EM iterations for UBM training
    UBM_N_INIT = 3              # Number of random initializations (pick best)
    
    # MAP Adaptation Parameters  
    MAP_RELEVANCE_FACTOR = 16.0  # Controls how much adaptation vs. UBM prior
                                  # Higher = more UBM influence (conservative)
                                  # Lower  = more speaker data influence (aggressive)
                                  # Typical range: 8-20
    MAP_ADAPT_WEIGHTS = True     # Adapt mixture weights
    MAP_ADAPT_MEANS = True       # Adapt means (most important!)
    MAP_ADAPT_COVARS = False     # Adapt covariances (usually False for small data)
    
    # Feature Processing
    USE_FEATURE_SCALING = True   # Standardize features (zero mean, unit variance)
    FEATURE_DIM = 39             # 13 MFCC + 13 Delta + 13 Delta-Delta


# =========================================================================
#  DATA LOADING
# =========================================================================
def load_mfcc_features(data_dir, set_name):
    """
    Load all MFCC .mat files from a directory.
    
    Returns:
        speakers_data: dict {speaker_id: features_matrix (num_frames x 39)}
    """
    mat_dir = os.path.join(data_dir, set_name)
    mat_files = sorted(glob.glob(os.path.join(mat_dir, '*_mfcc.mat')))
    
    if len(mat_files) == 0:
        raise FileNotFoundError(
            f"No .mat files found in '{mat_dir}'.\n"
            f"Run 'export_mfcc_features.m' in MATLAB first!"
        )
    
    speakers_data = {}
    for mat_file in mat_files:
        data = sio.loadmat(mat_file)
        speaker_id = str(data['speaker_id'][0]) if isinstance(data['speaker_id'], np.ndarray) else str(data['speaker_id'])
        
        # MATLAB saves as (coeffs × frames), e.g. mfcc_coeffs is (13 × num_frames)
        # Python needs (num_frames × 39), so transpose individual parts and hstack
        if 'mfcc_coeffs' in data and 'delta_mfcc' in data and 'delta2_mfcc' in data:
            mfcc_c = data['mfcc_coeffs'].T      # (num_frames × 13)
            delta_c = data['delta_mfcc'].T       # (num_frames × 13)
            delta2_c = data['delta2_mfcc'].T     # (num_frames × 13)
            features = np.hstack([mfcc_c, delta_c, delta2_c])  # (num_frames × 39)
        elif 'mfcc_coeffs' in data and 'delta_coeffs' in data and 'delta2_coeffs' in data:
            mfcc_c = data['mfcc_coeffs'].T
            delta_c = data['delta_coeffs'].T
            delta2_c = data['delta2_coeffs'].T
            features = np.hstack([mfcc_c, delta_c, delta2_c])
        else:
            # Fallback: use features directly, transpose if needed
            features = data['features']
            if features.shape[0] < features.shape[1]:
                features = features.T
        
        speakers_data[speaker_id] = features
        
    return speakers_data


def print_data_summary(train_data, test_data):
    """Display a summary of loaded data."""
    print("\n" + "=" * 55)
    print("  DATA SUMMARY")
    print("=" * 55)
    print(f"{'Speaker':<10} {'Train Frames':<15} {'Test Frames':<15} {'Dim':<5}")
    print("-" * 55)
    
    all_speakers = sorted(set(list(train_data.keys()) + list(test_data.keys())))
    total_train = 0
    total_test = 0
    
    for spk in all_speakers:
        tr = train_data.get(spk, np.array([]))
        te = test_data.get(spk, np.array([]))
        tr_frames = tr.shape[0] if len(tr.shape) > 1 else 0
        te_frames = te.shape[0] if len(te.shape) > 1 else 0
        dim = tr.shape[1] if len(tr.shape) > 1 else (te.shape[1] if len(te.shape) > 1 else 0)
        total_train += tr_frames
        total_test += te_frames
        print(f"{spk:<10} {tr_frames:<15} {te_frames:<15} {dim:<5}")
    
    print("-" * 55)
    print(f"{'TOTAL':<10} {total_train:<15} {total_test:<15}")
    print("=" * 55 + "\n")


# =========================================================================
#  GMM-UBM TRAINING
# =========================================================================
class GMMUBMSystem:
    """
    GMM-UBM Speaker Recognition System.
    
    The GMM-UBM approach:
    1. Train a Universal Background Model (UBM) on ALL speakers' data
       → This captures the general distribution of speech features
    2. For each speaker, adapt the UBM using MAP adaptation
       → This creates a speaker-specific GMM that captures what's unique
    3. At test time, compute log-likelihood ratio:
       score = log P(X | speaker_GMM) - log P(X | UBM)
       → Positive score = more like the specific speaker than general speech
    """
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.ubm = None                  # Universal Background Model
        self.speaker_gmms = {}           # {speaker_id: adapted_GMM}
        self.scaler = None               # Feature scaler
        self.speaker_list = []           # Ordered list of speaker IDs
        
    def _scale_features(self, features, fit=False):
        """Standardize features to zero mean and unit variance."""
        if not self.config.USE_FEATURE_SCALING:
            return features
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)
    
    # -----------------------------------------------------------------
    #  STEP 1: Train Universal Background Model
    # -----------------------------------------------------------------
    def train_ubm(self, all_train_data):
        """
        Train the UBM on pooled data from ALL speakers.
        
        The UBM represents "what speech in general sounds like".
        We pool all speakers' training data together.
        """
        print("[Step 1/2] Training Universal Background Model (UBM)...")
        print(f"  Components: {self.config.UBM_N_COMPONENTS}")
        print(f"  Covariance: {self.config.COVARIANCE_TYPE}")
        
        # Pool all training data
        all_features = []
        for spk_id, features in all_train_data.items():
            all_features.append(features)
        all_features = np.vstack(all_features)
        
        print(f"  Pooled data: {all_features.shape[0]} frames × {all_features.shape[1]} features")
        
        # Scale features
        all_features_scaled = self._scale_features(all_features, fit=True)
        
        # Train UBM using EM algorithm
        self.ubm = GaussianMixture(
            n_components=self.config.UBM_N_COMPONENTS,
            covariance_type=self.config.COVARIANCE_TYPE,
            max_iter=self.config.UBM_MAX_ITER,
            n_init=self.config.UBM_N_INIT,
            random_state=42,
            verbose=0
        )
        self.ubm.fit(all_features_scaled)
        
        print(f"  ✓ UBM trained! (converged: {self.ubm.converged_})")
        print(f"  ✓ Log-likelihood on training data: {self.ubm.score(all_features_scaled):.2f}\n")
        
        return self.ubm
    
    # -----------------------------------------------------------------
    #  STEP 2: MAP Adaptation for each speaker
    # -----------------------------------------------------------------
    def adapt_speaker_gmm(self, speaker_id, speaker_features):
        """
        Adapt the UBM to a specific speaker using MAP adaptation.
        
        MAP (Maximum A Posteriori) adaptation:
        - Start from UBM parameters (prior)
        - Update towards speaker's data
        - Relevance factor controls the balance
        
        This is more robust than training a GMM from scratch per speaker,
        especially with limited data.
        
        Formula for mean adaptation:
            adapted_mean_i = α_i * speaker_mean_i + (1 - α_i) * ubm_mean_i
            where α_i = n_i / (n_i + relevance_factor)
            and n_i = soft count of frames assigned to component i
        """
        # Scale speaker features
        speaker_features_scaled = self._scale_features(speaker_features)
        
        # E-step: compute posterior probabilities (responsibilities)
        # For each frame, how likely is it to belong to each Gaussian component?
        posteriors = self.ubm.predict_proba(speaker_features_scaled)  # (num_frames x num_components)
        
        # Compute soft counts per component
        n_k = posteriors.sum(axis=0)  # (num_components,) — effective # frames per component
        
        # Compute first-order statistics (weighted sums)
        # For each component k: sum of (posterior * feature) across all frames
        F_k = posteriors.T @ speaker_features_scaled  # (num_components x feature_dim)
        
        # Create adapted GMM (start as a copy of UBM)
        adapted_gmm = GaussianMixture(
            n_components=self.config.UBM_N_COMPONENTS,
            covariance_type=self.config.COVARIANCE_TYPE,
        )
        # Copy UBM parameters
        adapted_gmm.means_ = self.ubm.means_.copy()
        adapted_gmm.covariances_ = self.ubm.covariances_.copy()
        adapted_gmm.weights_ = self.ubm.weights_.copy()
        adapted_gmm.precisions_cholesky_ = self.ubm.precisions_cholesky_.copy()
        adapted_gmm.converged_ = True
        
        # MAP adaptation
        r = self.config.MAP_RELEVANCE_FACTOR
        
        for k in range(self.config.UBM_N_COMPONENTS):
            # Adaptation coefficient (how much to trust speaker data vs UBM)
            alpha_k = n_k[k] / (n_k[k] + r)
            
            # Adapt means (most important for speaker recognition)
            if self.config.MAP_ADAPT_MEANS and n_k[k] > 0:
                speaker_mean_k = F_k[k] / n_k[k]  # Weighted mean of speaker data for component k
                adapted_gmm.means_[k] = alpha_k * speaker_mean_k + (1 - alpha_k) * self.ubm.means_[k]
            
            # Adapt weights
            if self.config.MAP_ADAPT_WEIGHTS and n_k[k] > 0:
                adapted_gmm.weights_[k] = alpha_k * (n_k[k] / len(speaker_features_scaled)) + \
                                           (1 - alpha_k) * self.ubm.weights_[k]
        
        # Normalize weights to sum to 1
        if self.config.MAP_ADAPT_WEIGHTS:
            adapted_gmm.weights_ /= adapted_gmm.weights_.sum()
        
        # Recompute precision (inverse covariance) for scoring
        if self.config.COVARIANCE_TYPE == 'diag':
            adapted_gmm.precisions_cholesky_ = 1.0 / np.sqrt(adapted_gmm.covariances_)
        
        self.speaker_gmms[speaker_id] = adapted_gmm
        return adapted_gmm
    
    def train(self, train_data):
        """Full training pipeline: UBM + MAP adaptation for all speakers."""
        self.speaker_list = sorted(train_data.keys())
        
        # Step 1: Train UBM
        self.train_ubm(train_data)
        
        # Step 2: Adapt for each speaker
        print("[Step 2/2] MAP Adaptation for each speaker...")
        for spk_id in self.speaker_list:
            features = train_data[spk_id]
            self.adapt_speaker_gmm(spk_id, features)
            print(f"  ✓ Adapted GMM for speaker '{spk_id}' ({features.shape[0]} frames)")
        
        print(f"\n  ✓ All {len(self.speaker_list)} speaker models trained!\n")
    
    # -----------------------------------------------------------------
    #  SCORING & IDENTIFICATION
    # -----------------------------------------------------------------
    def score_utterance(self, features):
        """
        Score a test utterance against all speaker models.
        
        For each speaker, compute:
            score = (1/T) * [log P(X | speaker_GMM) - log P(X | UBM)]
        
        This is the log-likelihood ratio (LLR), normalized by number of frames.
        Positive = more like the speaker than general speech.
        
        Returns:
            scores: dict {speaker_id: score}
        """
        features_scaled = self._scale_features(features)
        T = len(features_scaled)  # Number of frames
        
        # UBM score (baseline)
        ubm_score = self.ubm.score(features_scaled)  # Average log-likelihood
        
        scores = {}
        for spk_id, gmm in self.speaker_gmms.items():
            spk_score = gmm.score(features_scaled)  # Average log-likelihood
            # Log-likelihood ratio
            scores[spk_id] = spk_score - ubm_score
        
        return scores
    
    def identify(self, features, threshold=0.0):
        """
        Identify the speaker of a test utterance.
        
        Returns:
            predicted_speaker: speaker ID with highest score
            confidence: confidence percentage (softmax over scores)
            all_scores: dict of all speaker scores
        """
        scores = self.score_utterance(features)
        
        # Find best speaker
        predicted_speaker = max(scores, key=scores.get)
        best_score = scores[predicted_speaker]
        
        # Convert scores to confidence using softmax
        score_values = np.array([scores[spk] for spk in self.speaker_list])
        
        # Softmax for confidence (temperature-scaled)
        temperature = 1.0
        exp_scores = np.exp((score_values - score_values.max()) / temperature)
        confidences = exp_scores / exp_scores.sum()
        
        confidence_dict = {spk: conf for spk, conf in zip(self.speaker_list, confidences)}
        
        return predicted_speaker, confidence_dict[predicted_speaker], confidence_dict, scores
    
    # -----------------------------------------------------------------
    #  SAVE / LOAD MODELS
    # -----------------------------------------------------------------
    def save_models(self, model_dir=None):
        """Save all trained models to disk."""
        model_dir = model_dir or self.config.MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        
        model_data = {
            'ubm': self.ubm,
            'speaker_gmms': self.speaker_gmms,
            'scaler': self.scaler,
            'speaker_list': self.speaker_list,
            'config': self.config
        }
        
        model_path = os.path.join(model_dir, 'gmm_ubm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  ✓ Models saved to '{model_path}'")
        return model_path
    
    def load_models(self, model_dir=None):
        """Load trained models from disk."""
        model_dir = model_dir or self.config.MODEL_DIR
        model_path = os.path.join(model_dir, 'gmm_ubm_model.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ubm = model_data['ubm']
        self.speaker_gmms = model_data['speaker_gmms']
        self.scaler = model_data['scaler']
        self.speaker_list = model_data['speaker_list']
        self.config = model_data['config']
        
        print(f"  ✓ Models loaded from '{model_path}'")
        print(f"  ✓ {len(self.speaker_list)} speakers: {self.speaker_list}")
        return self


# =========================================================================
#  EVALUATION
# =========================================================================
def evaluate(system, test_data):
    """
    Evaluate the system on test data.
    
    For each test utterance, identify the speaker and check if correct.
    """
    print("=" * 60)
    print("  EVALUATION ON TEST SET")
    print("=" * 60)
    
    correct = 0
    total = 0
    results = []
    
    for true_speaker, features in sorted(test_data.items()):
        predicted, confidence, all_conf, scores = system.identify(features)
        is_correct = (predicted == true_speaker)
        correct += int(is_correct)
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} True: {true_speaker:<6} → Predicted: {predicted:<6} "
              f"(confidence: {confidence:.1%})")
        
        # Show top-3 candidates
        sorted_conf = sorted(all_conf.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_conf[:3]
        details = " | ".join([f"{spk}: {conf:.1%}" for spk, conf in top3])
        print(f"    Top-3: {details}")
        
        results.append({
            'true_speaker': true_speaker,
            'predicted_speaker': predicted,
            'confidence': confidence,
            'correct': is_correct,
            'all_confidences': all_conf,
            'scores': scores
        })
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"  ACCURACY: {correct}/{total} = {accuracy:.1%}")
    print(f"{'=' * 60}\n")
    
    return results, accuracy


def print_confusion_info(results, speaker_list):
    """Print a simple confusion-style summary."""
    print("DETAILED SCORE MATRIX (Log-Likelihood Ratios):")
    print("-" * 60)
    
    header = f"{'True↓ / Pred→':<12}" + "".join([f"{spk:<8}" for spk in speaker_list])
    print(header)
    print("-" * 60)
    
    for result in results:
        true_spk = result['true_speaker']
        row = f"{true_spk:<12}"
        for spk in speaker_list:
            score = result['scores'].get(spk, 0)
            if spk == result['predicted_speaker']:
                row += f"[{score:>5.2f}] "
            else:
                row += f" {score:>5.2f}  "
        print(row)
    print("-" * 60)
    print("[ ] = predicted speaker\n")


# =========================================================================
#  MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='GMM-UBM Speaker Recognition')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Load saved models and evaluate only')
    parser.add_argument('--components', type=int, default=16,
                        help='Number of GMM components (default: 16)')
    parser.add_argument('--relevance', type=float, default=16.0,
                        help='MAP relevance factor (default: 16.0)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  GMM-UBM SPEAKER RECOGNITION SYSTEM")
    print("=" * 60 + "\n")
    
    # Update config
    config = Config()
    config.UBM_N_COMPONENTS = args.components
    config.MAP_RELEVANCE_FACTOR = args.relevance
    
    # Load data
    print("Loading MFCC features from MATLAB export...")
    train_data = load_mfcc_features(config.EXPORTED_DIR, 'train')
    test_data = load_mfcc_features(config.EXPORTED_DIR, 'test')
    print_data_summary(train_data, test_data)
    
    # Create system
    system = GMMUBMSystem(config)
    
    if args.evaluate:
        # Load existing model
        print("Loading saved models...")
        system.load_models()
    else:
        # Train
        system.train(train_data)
        system.save_models()
    
    # Evaluate
    results, accuracy = evaluate(system, test_data)
    print_confusion_info(results, system.speaker_list)
    
    return system, results, accuracy


if __name__ == '__main__':
    main()

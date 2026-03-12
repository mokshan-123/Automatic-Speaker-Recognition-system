function [features, mfcc_coeffs, delta_coeffs, delta2_coeffs] = mfcc(signal, fs, show_progress)
% MFCC - Extract 39-dimensional MFCC features (13 MFCC + 13 Delta + 13 Delta-Delta)
%
% This is the MAIN function that orchestrates the entire MFCC pipeline.
% It calls 8 helper functions in sequence to transform raw audio into
% complete MFCC features suitable for speaker recognition.
%
% INPUTS:
%   signal        - input audio waveform (column vector or row vector)
%   fs            - sampling frequency in Hz (e.g., 8000, 16000, 22050, 44100)
%   show_progress - (optional) true to display progress, false for silent (default: true)
%
% OUTPUTS:
%   features      - 39-dimensional feature matrix (num_frames × 39)
%                   [13 MFCC | 13 Delta | 13 Delta-Delta]
%   mfcc_coeffs   - MFCC coefficients only (num_frames × 13)
%   delta_coeffs  - Delta coefficients (num_frames × 13)
%   delta2_coeffs - Delta-delta coefficients (num_frames × 13)
%
% PIPELINE:
%   1. Pre-emphasis       → boost high frequencies
%   2. Frame blocking     → divide into overlapping segments
%   3. Windowing          → apply Hamming window
%   4. FFT                → time domain → frequency domain
%   5. Power spectrum     → compute energy at each frequency
%   6. Mel filterbank     → perceptual frequency warping
%   7. Logarithm          → compress dynamic range
%   8. DCT                → decorrelate and compress
%
% USAGE:
%   [audio, fs] = audioread('speech.wav');
%   features = mfcc(audio, fs);  % Returns 39-dimensional features
%
%   % Get individual components:
%   [features, mfcc, delta, delta2] = mfcc(audio, fs);
%
%   % Silent mode (no progress output):
%   features = mfcc(audio, fs, false);
%
% EXAMPLE (Complete workflow):
%   % Load speech
%   [audio, fs] = audioread('S1.wav');
%   
%   % Extract 39-dimensional MFCC features with progress
%   [features, mfcc, delta, delta2] = mfcc(audio, fs, true);
%   
%   % Display info
%   fprintf('Output: %d frames × %d features\n', size(features));
%   fprintf('  - MFCC:        columns 1-13\n');
%   fprintf('  - Delta:       columns 14-26\n');
%   fprintf('  - Delta-Delta: columns 27-39\n');
%   
%   % Visualize all 39 features
%   figure;
%   imagesc(features'); axis xy; colorbar; colormap('jet');
%   xlabel('Frame'); ylabel('Feature Dimension');
%   title('39-Dimensional MFCC Features');
%   hold on;
%   plot([0.5 size(features,1)+0.5], [13.5 13.5], 'w--', 'LineWidth', 2);
%   plot([0.5 size(features,1)+0.5], [26.5 26.5], 'w--', 'LineWidth', 2);
%   hold off;
%
% PARAMETERS (adjust as needed for your project):
%   alpha        = 0.97     Pre-emphasis coefficient
%   frame_length = 256      Frame size in samples (~16 ms at 16 kHz)
%   hop_size     = 100      Frame shift (~6 ms at 16 kHz, 61% overlap)
%   nfft         = 512      FFT size (zero-pads to this length)
%   num_filters  = 26       Number of mel filterbank filters
%   num_coeffs   = 13       Number of MFCC coefficients to extract
%
% REQUIREMENTS:
%   This function requires these helper functions in the same directory:
%   - preemphasis.m
%   - frame_signal.m
%   - apply_window.m
%   - compute_power_spectrum.m
%   - apply_mel_filterbank.m
%   - apply_log.m
%   - apply_dct.m
%   - compute_delta.m
%   
%   Plus the provided function:
%   - melfb.m (creates mel filterbank)

% =========================================================================
% STEP 0: INPUT VALIDATION
% =========================================================================
if nargin < 2
    error('MFCC:NotEnoughInputs', ...
          ['MFCC requires at least 2 input arguments: mfcc(signal, fs)\n\n' ...
           'USAGE:\n' ...
           '  [audio, fs] = audioread(''speech.wav'');\n' ...
           '  features = mfcc(audio, fs);\n' ...
           '  [features, mfcc, delta, delta2] = mfcc(audio, fs, true);\n\n' ...
           'Type ''help mfcc'' for more information.']);
end

if nargin < 3
    show_progress = true;  % Default: show progress
end

% =========================================================================
% STEP 0.5: CONFIGURE PARAMETERS
% =========================================================================
% These are the standard values. Adjust if your project requires different settings.

alpha        = 0.97;    % Pre-emphasis coefficient (0.95-0.97 typical)
frame_length = 256;     % Samples per frame (20-30 ms typical)
                        % At fs=16kHz: 256 samples = 16 ms
                        % At fs=8kHz:  256 samples = 32 ms
hop_size     = 100;     % Frame shift in samples (10 ms typical)
                        % At fs=16kHz: 100 samples = 6.25 ms
                        % Overlap = frame_length - hop_size = 156 samples
nfft         = 512;     % FFT size (power of 2, >= frame_length)
num_filters  = 26;      % Mel filterbank filters (20-40 typical)
num_coeffs   = 13;      % MFCC coefficients to keep (12-13 standard)

% NEW OPTIONS:
drop_c0      = false;   % Set to true to drop the first coefficient (c0/energy)
                        % c0 is loudness-dependent and may not carry
                        % speaker-specific information. Some systems drop it.
delta_N      = 2;       % Window size for delta computation (frames on each side)

% =========================================================================
% STEP 0.5: INPUT VALIDATION AND PREPROCESSING
% =========================================================================
% Ensure signal is a column vector
if isrow(signal)
    signal = signal(:);
end

% Check signal length
if length(signal) < frame_length
    error('Signal too short! Need at least %d samples, got %d.', ...
          frame_length, length(signal));
end

% Optional: Remove DC offset (mean)
% Uncomment if your audio has a DC component:
% signal = signal - mean(signal);

if show_progress
    fprintf('\n========================================\n');
    fprintf('  MFCC FEATURE EXTRACTION PIPELINE\n');
    fprintf('========================================\n');
    fprintf('Signal: %d samples (%.2f sec @ %d Hz)\n\n', ...
            length(signal), length(signal)/fs, fs);
end

% =========================================================================
% STEP 1: PRE-EMPHASIS
% =========================================================================
% Boost high frequencies to balance the spectrum
if show_progress
    fprintf('[Step 1/8] Pre-emphasis filter...\n');
end
emphasized = preemphasis(signal, alpha);
if show_progress
    fprintf('  ✓ Applied pre-emphasis (α=%.2f)\n\n', alpha);
end

% =========================================================================
% STEP 2: FRAME BLOCKING
% =========================================================================
% Divide signal into overlapping frames
% Output: (num_frames × frame_length) matrix, each row = one frame
if show_progress
    fprintf('[Step 2/8] Frame blocking...\n');
end
frames = frame_signal(emphasized, frame_length, hop_size);
if show_progress
    fprintf('  ✓ Created %d frames (%d samples, %d hop, %.1f%% overlap)\n\n', ...
            size(frames,1), frame_length, hop_size, ...
            (frame_length-hop_size)/frame_length*100);
end

% =========================================================================
% STEP 3: WINDOWING
% =========================================================================
% Apply Hamming window to each frame to reduce spectral leakage
% Output: same size as frames
if show_progress
    fprintf('[Step 3/8] Applying Hamming window...\n');
end
windowed_frames = apply_window(frames);
if show_progress
    fprintf('  ✓ Windowed %d frames\n\n', size(windowed_frames,1));
end

% =========================================================================
% STEP 4 & 5: FFT + POWER SPECTRUM
% =========================================================================
% Compute FFT and then power spectrum (|FFT|^2)
% Output: (num_frames × (nfft/2 + 1)) matrix
if show_progress
    fprintf('[Step 4/8] Computing FFT and power spectrum...\n');
end
power_spectrum = compute_power_spectrum(windowed_frames, nfft);
if show_progress
    fprintf('  ✓ Power spectrum: %d frames × %d bins (%.1f Hz resolution)\n\n', ...
            size(power_spectrum), fs/nfft);
end

% =========================================================================
% STEP 6: MEL FILTERBANK
% =========================================================================
% Apply mel-spaced triangular filterbank
% Uses the provided melfb.m function
% Output: (num_frames × num_filters) matrix
if show_progress
    fprintf('[Step 5/8] Applying mel filterbank...\n');
end
mel_energies = apply_mel_filterbank(power_spectrum, fs, nfft, num_filters);
if show_progress
    fprintf('  ✓ Mel energies: %d frames × %d filters\n\n', ...
            size(mel_energies));
end

% =========================================================================
% STEP 7: LOGARITHM
% =========================================================================
% Take log to compress dynamic range and match human perception
% Output: same size as mel_energies
if show_progress
    fprintf('[Step 6/8] Applying logarithm...\n');
end
log_mel = apply_log(mel_energies);
if show_progress
    fprintf('  ✓ Log-mel energies computed\n\n');
end

% =========================================================================
% STEP 8: DCT
% =========================================================================
% Apply Discrete Cosine Transform and keep first num_coeffs
% Output: (num_frames × num_coeffs) matrix by default
if show_progress
    fprintf('[Step 7/8] Applying DCT...\n');
end
mfcc_coeffs = apply_dct(log_mel, num_coeffs);
if show_progress
    fprintf('  ✓ MFCC coefficients: %d frames × %d coeffs\n\n', ...
            size(mfcc_coeffs));
end

% =========================================================================
% STEP 8.5: OPTIONAL - DROP C0 (ENERGY COEFFICIENT)
% =========================================================================
% The first coefficient (c0) is proportional to log-energy/loudness.
% Some systems drop it because it's volume-dependent, not speaker-dependent.
if drop_c0
    mfcc_coeffs = mfcc_coeffs(:, 2:end);  % Remove first column (c0)
    if show_progress
        fprintf('  ⚠ Dropped c0 coefficient (energy term)\n\n');
    end
end

% =========================================================================
% STEP 9: COMPUTE DELTA AND DELTA-DELTA FEATURES
% =========================================================================
% Delta features capture temporal dynamics (how MFCCs change over time).
% Delta-delta features capture acceleration (second derivative).
% Standard: 13 MFCC + 13 delta + 13 delta-delta = 39 features
if show_progress
    fprintf('[Step 8/8] Computing delta features...\n');
end

delta_coeffs = compute_delta(mfcc_coeffs, delta_N);        % First derivative (velocity)
delta2_coeffs = compute_delta(delta_coeffs, delta_N);      % Second derivative (acceleration)

if show_progress
    fprintf('  ✓ Delta coefficients: %d frames × %d coeffs\n', ...
            size(delta_coeffs));
    fprintf('  ✓ Delta-delta coefficients: %d frames × %d coeffs\n\n', ...
            size(delta2_coeffs));
end

% Concatenate to form 39-dimensional feature vector
features = [mfcc_coeffs, delta_coeffs, delta2_coeffs];

if show_progress
    fprintf('========================================\n');
    fprintf('  EXTRACTION COMPLETE!\n');
    fprintf('========================================\n');
    fprintf('Final output: %d frames × %d features\n', size(features));
    fprintf('  • MFCC:        columns 1-%d\n', size(mfcc_coeffs,2));
    fprintf('  • Delta:       columns %d-%d\n', size(mfcc_coeffs,2)+1, size(mfcc_coeffs,2)*2);
    fprintf('  • Delta-Delta: columns %d-%d\n', size(mfcc_coeffs,2)*2+1, size(features,2));
    fprintf('\nFeature statistics:\n');
    fprintf('  Range: [%.2f, %.2f]\n', min(features(:)), max(features(:)));
    fprintf('  Mean:  %.4f\n', mean(features(:)));
    fprintf('  Std:   %.4f\n', std(features(:)));
    fprintf('========================================\n\n');
end

end

% =========================================================================
% FUNCTION COMPLETE - 39-DIMENSIONAL MFCC FEATURES READY!
% =========================================================================

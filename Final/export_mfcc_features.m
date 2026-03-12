% EXPORT_MFCC_FEATURES - Batch export MFCC features for all speakers
%
% Uses the SAME step-by-step pipeline as our MLX notebook.
% All audio is truncated to the SHORTEST recording so every speaker
% produces the same number of frames → consistent matrix sizes.
%
% Each .mat file contains:
%   mfcc_coeffs  - (13 × num_frames)
%   delta_mfcc   - (13 × num_frames)
%   delta2_mfcc  - (13 × num_frames)
%   features     - (39 × num_frames)  [mfcc; delta; delta-delta]
%   speaker_id   - string label
%   fs           - sampling frequency

clear; clc;

fprintf('==============================================\n');
fprintf('  BATCH MFCC EXPORT FOR SPEAKER RECOGNITION\n');
fprintf('==============================================\n\n');

% =========================================================================
%  PARAMETERS (same as MLX notebook)
% =========================================================================
alpha        = 0.97;
frame_length = 256;
hop_size     = 100;
nfft         = 512;
num_filters  = 26;
num_coeffs   = 13;

% =========================================================================
%  STEP 0: Find minimum audio length across ALL files (train + test)
% =========================================================================
sets = {'train', 'test'};
min_len = Inf;

fprintf('[Step 0] Scanning all audio files for minimum length...\n');
for s = 1:length(sets)
    wav_files = dir(fullfile('data', sets{s}, '*.wav'));
    for i = 1:length(wav_files)
        wav_path = fullfile('data', sets{s}, wav_files(i).name);
        [audio, fs] = audioread(wav_path);
        if length(audio) < min_len
            min_len = length(audio);
        end
        fprintf('  %s/%s: %d samples (%.2f sec)\n', ...
                sets{s}, wav_files(i).name, length(audio), length(audio)/fs);
    end
end

fprintf('\n  → Minimum length: %d samples (%.2f sec @ %d Hz)\n', min_len, min_len/fs, fs);
fprintf('  → All audio will be truncated to %d samples\n\n', min_len);

% =========================================================================
%  STEP 1: Process and export
% =========================================================================
output_base = 'exported_features';
if ~exist(output_base, 'dir')
    mkdir(output_base);
end

for s = 1:length(sets)
    set_name = sets{s};
    input_dir = fullfile('data', set_name);
    output_dir = fullfile(output_base, set_name);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    wav_files = dir(fullfile(input_dir, '*.wav'));

    fprintf('Processing %s set (%d files)...\n', upper(set_name), length(wav_files));
    fprintf('------------------------------------------\n');

    for i = 1:length(wav_files)
        % Load audio
        wav_path = fullfile(input_dir, wav_files(i).name);
        [audio, fs] = audioread(wav_path);

        % Truncate to minimum length
        audio = audio(1:min_len);

        % Extract speaker ID from filename
        [~, speaker_id] = fileparts(wav_files(i).name);

        % === MFCC PIPELINE (same as MLX notebook) ===

        % Step 1: Pre-emphasis
        emphasized = preemphasis(audio, alpha);

        % Step 2: Frame blocking
        frames = frame_signal(emphasized, frame_length, hop_size);

        % Step 3: Windowing
        windowed_frames = apply_window(frames);

        % Step 4 & 5: FFT + Power spectrum
        power_spectrum = compute_power_spectrum(windowed_frames, nfft);

        % Step 6: Mel filterbank
        mel_energies = apply_mel_filterbank(power_spectrum, fs, nfft, num_filters);

        % Step 7: Logarithm
        log_mel = apply_log(mel_energies);

        % Step 8: DCT → 13 MFCC coefficients
        mfcc_coeffs = apply_dct(log_mel, num_coeffs);

        % Step 9: Delta and Delta-Delta
        delta_mfcc = compute_delta(mfcc_coeffs);
        delta2_mfcc = compute_delta(delta_mfcc);

        % Step 10: Concatenate → 39-dimensional feature vector
        % Result: (39 × num_frames) — same orientation as MLX notebook
        features = [mfcc_coeffs; delta_mfcc; delta2_mfcc];

        % Save to .mat file
        output_file = fullfile(output_dir, [speaker_id '_mfcc.mat']);
        save(output_file, 'features', 'mfcc_coeffs', 'delta_mfcc', 'delta2_mfcc', ...
             'speaker_id', 'fs');

        fprintf('  [%d/%d] %s → %d features × %d frames → %s\n', ...
                i, length(wav_files), wav_files(i).name, ...
                size(features, 1), size(features, 2), output_file);
    end
    fprintf('\n');
end

% =========================================================================
%  Save metadata
% =========================================================================
metadata.speakers = {};
train_files = dir(fullfile('data', 'train', '*.wav'));
for i = 1:length(train_files)
    [~, sid] = fileparts(train_files(i).name);
    metadata.speakers{end+1} = sid;
end
metadata.num_speakers = length(metadata.speakers);
metadata.feature_dim = 39;
metadata.min_audio_length = min_len;
metadata.fs = fs;
metadata.feature_description = '13 MFCC + 13 Delta + 13 Delta-Delta';
metadata.parameters.alpha = alpha;
metadata.parameters.frame_length = frame_length;
metadata.parameters.hop_size = hop_size;
metadata.parameters.nfft = nfft;
metadata.parameters.num_filters = num_filters;
metadata.parameters.num_coeffs = num_coeffs;
save(fullfile(output_base, 'metadata.mat'), 'metadata');

fprintf('==============================================\n');
fprintf('  EXPORT COMPLETE!\n');
fprintf('==============================================\n');
fprintf('Output directory: %s\n', fullfile(pwd, output_base));
fprintf('Speakers: %d\n', metadata.num_speakers);
fprintf('Feature dimension: %d\n', metadata.feature_dim);
fprintf('Audio truncated to: %d samples (%.2f sec)\n', min_len, min_len/fs);
fprintf('\nNext step: Run the Python training pipeline!\n');
fprintf('  > python train_gmm_ubm.py\n');
fprintf('==============================================\n');

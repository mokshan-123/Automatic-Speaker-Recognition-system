%% ============================================
%% CONFIGURATION
%% ============================================

% Set your base folder path (folder containing speaker folders)
base_folder = 'D:\Individual projects\An-Automatic-Speaker-Recognition-System-heshan\Voices';  % CHANGE THIS

% Audio processing parameters
frame_size = 512;        % Samples per frame
overlap = 256;           % Overlap between frames
FFT_len = 512;          % FFT length
num_melbank = 40;       % Number of Mel filters

%% ============================================
%% PROCESS ALL SPEAKERS
%% ============================================

% Get all speaker folders
speaker_folders = dir(base_folder);
speaker_folders = speaker_folders([speaker_folders.isdir]);
speaker_folders = speaker_folders(~ismember({speaker_folders.name}, {'.', '..'}));

fprintf('Found %d speaker folders\n\n', length(speaker_folders));

% Initialize storage
all_features = [];
all_labels = {};

% Process each speaker folder
for s = 1:length(speaker_folders)
    speaker_name = speaker_folders(s).name;
    speaker_path = fullfile(base_folder, speaker_name);
    
    % Try to find audio files with different extensions
    audio_files = [];
    extensions = {'*.wav', '*.WAV', '*.mp3', '*.MP3', '*.flac', '*.FLAC'};
    
    for ext_idx = 1:length(extensions)
        found_files = dir(fullfile(speaker_path, extensions{ext_idx}));
        audio_files = [audio_files; found_files];
    end
    
    % Also check in subfolders (one level deep)
    subfolders = dir(speaker_path);
    subfolders = subfolders([subfolders.isdir]);
    subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
    
    for sf = 1:length(subfolders)
        subfolder_path = fullfile(speaker_path, subfolders(sf).name);
        for ext_idx = 1:length(extensions)
            found_files = dir(fullfile(subfolder_path, extensions{ext_idx}));
            % Add subfolder name to the file structure
            for f = 1:length(found_files)
                found_files(f).folder = subfolder_path;
            end
            audio_files = [audio_files; found_files];
        end
    end
    
    fprintf('Processing %s: %d audio files\n', speaker_name, length(audio_files));
    
    if length(audio_files) == 0
        fprintf('  WARNING: No audio files found in %s\n', speaker_path);
        fprintf('  Please check if files exist and what extension they have.\n\n');
        continue;
    end
    % Create progress bar
    h = waitbar(0, sprintf('Processing %s (0/%d)', speaker_name, length(audio_files)));
    % Process each audio file
    for a = 1:length(audio_files)
        audio_file = audio_files(a).name;
        audio_path = fullfile(audio_files(a).folder, audio_file);
        % Update progress bar
        waitbar(a/length(audio_files), h, sprintf('Processing %s (%d/%d)', speaker_name, a, length(audio_files)));
        
        try
            % Load audio
            [audio_signal, fs] = audioread(audio_path);
            
            % Convert stereo to mono if needed
            if size(audio_signal, 2) > 1
                audio_signal = mean(audio_signal, 2);
            end
            
            % Buffer audio into frames
            frames = buffer(audio_signal, frame_size, overlap);
            
            % Call YOUR MFCC function
            [~, ~, ~, ~, Final_feature_vector] = get_FFTP_Melbank_MFCC_Featurevector(...
                frames, fs, FFT_len, num_melbank);
            
            % Convert variable-length features to fixed-size
            % Final_feature_vector shape: (39, num_frames)
            feature_mean = mean(Final_feature_vector, 2)';  % (1, 39)
            feature_std = std(Final_feature_vector, 0, 2)'; % (1, 39)
            feature_min = min(Final_feature_vector, [], 2)';% (1, 39)
            feature_max = max(Final_feature_vector, [], 2)';% (1, 39)
            
            % Concatenate all statistics
            final_features = [feature_mean, feature_std, feature_min, feature_max];
            % Shape: (1, 156) = 39 features × 4 statistics
            
            % Store features and label
            all_features = [all_features; final_features];
            all_labels{end+1} = speaker_name;
            
            if mod(a, 10) == 0
                fprintf('  Processed %d/%d files\n', a, length(audio_files));
            end
            
        catch ME
            fprintf('  ERROR processing %s: %s\n', audio_file, ME.message);
            continue;
        end
    end
    % Close progress bar
    close(h);
    fprintf('  Completed %s\n\n', speaker_name);
end

%% ============================================
%% CHECK IF WE HAVE DATA
%% ============================================

if isempty(all_features)
    error('No audio files were processed! Please check:\n1. Folder path is correct\n2. Audio files exist\n3. File extensions (.wav, .mp3, .FLAC, etc.)');
end

%% ============================================
%% CREATE DATASET TABLE
%% ============================================

fprintf('\n========================================\n');
fprintf('Dataset Creation Complete!\n');
fprintf('========================================\n');
fprintf('Total samples: %d\n', size(all_features, 1));
fprintf('Feature dimension: %d\n', size(all_features, 2));
fprintf('Unique speakers: %d\n', length(unique(all_labels)));

% Create column names
% 39 features (13 MFCC + 13 delta + 13 delta-delta) × 4 statistics = 156 columns
feature_columns = {};
feature_types = {'mfcc', 'delta_mfcc', 'delta_delta_mfcc'};
stats = {'mean', 'std', 'min', 'max'};

for st = 1:length(stats)
    for ft = 1:length(feature_types)
        for i = 0:12
            col_name = sprintf('%s_%d_%s', feature_types{ft}, i, stats{st});
            feature_columns{end+1} = col_name;
        end
    end
end

% Debug: Check sizes match
fprintf('\nDebug Info:\n');
fprintf('Number of features (columns): %d\n', size(all_features, 2));
fprintf('Number of column names: %d\n', length(feature_columns));

% Create table
if size(all_features, 2) == length(feature_columns)
    T = array2table(all_features, 'VariableNames', feature_columns);
else
    fprintf('Warning: Column count mismatch. Using auto-generated names.\n');
    T = array2table(all_features);
end

T.speaker = all_labels';

%% ============================================
%% SAVE DATASET
%% ============================================

fprintf('\nSaving dataset...\n');

% Save to Excel
writetable(T, 'mfcc_dataset_11speakers.xlsx');
fprintf('  Saved: mfcc_dataset_11speakers.xlsx\n');

% Save to CSV
writetable(T, 'mfcc_dataset_11speakers.csv');
fprintf('  Saved: mfcc_dataset_11speakers.csv\n');

%% ============================================
%% DISPLAY SUMMARY
%% ============================================

fprintf('\nDataset Summary:\n');
fprintf('Shape: %d rows × %d columns\n', size(T, 1), size(T, 2));

fprintf('\nSamples per speaker:\n');
[unique_speakers, ~, idx] = unique(T.speaker);
for i = 1:length(unique_speakers)
    count = sum(idx == i);
    fprintf('  %s: %d samples\n', unique_speakers{i}, count);
end

fprintf('\nFirst few rows:\n');
disp(head(T));

fprintf('\nDataset creation completed successfully!\n');
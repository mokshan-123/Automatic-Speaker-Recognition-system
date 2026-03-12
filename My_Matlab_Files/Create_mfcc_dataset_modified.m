%% ============================================
%% CONFIGURATION
%% ============================================

% Set your base folder path (folder containing speaker folders)
base_folder = 'D:\Individual projects\An-Automatic-Speaker-Recognition-System-heshan\Voices';

% Audio processing parameters
frame_size = 512;        % Samples per frame
overlap = 256;           % Overlap between frames
FFT_len = 512;          % FFT length
num_melbank = 40;       % Number of Mel filters

%Apply Mel Banks
Melbank_filter=MelBank(fs,FFT_len,num_melbank);

% Create output folder for individual speaker files
output_folder = fullfile(base_folder, 'speaker_datasets');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ============================================
%% PROCESS ALL SPEAKERS
%% ============================================

% Get all speaker folders
speaker_folders = dir(base_folder);
speaker_folders = speaker_folders([speaker_folders.isdir]);
speaker_folders = speaker_folders(~ismember({speaker_folders.name}, {'.', '..', 'speaker_datasets'}));

fprintf('========================================\n');
fprintf('SPEAKER RECOGNITION DATASET CREATOR\n');
fprintf('========================================\n');
fprintf('Found %d speaker folders\n', length(speaker_folders));
fprintf('Output folder: %s\n\n', output_folder);

% Initialize storage for COMBINED dataset
all_features = [];
all_labels = {};

% Process each speaker folder
for s = 1:length(speaker_folders)
    speaker_name = speaker_folders(s).name;
    speaker_path = fullfile(base_folder, speaker_name);
    
    fprintf('Speaker %d/%d: %s\n', s, length(speaker_folders), speaker_name);
    
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
    
    fprintf('  Found %d audio files\n', length(audio_files));
    
    if length(audio_files) == 0
        fprintf('  WARNING: No audio files found!\n\n');
        continue;
    end
    
    % Initialize storage for THIS SPEAKER ONLY
    speaker_features = [];
    speaker_labels = {};
    
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
            [~, ~, ~, Final_feature_vector] = get_FFTP_Melbank_MFCC_Featurevector(...
                frames, fs, FFT_len, num_melbank,Melbank_filter);
            
            % Convert variable-length features to fixed-size
            % Final_feature_vector shape: (39, num_frames)
            feature_mean = mean(Final_feature_vector, 2)';  % (1, 39)
            feature_std = std(Final_feature_vector, 0, 2)'; % (1, 39)
            feature_min = min(Final_feature_vector, [], 2)';% (1, 39)
            feature_max = max(Final_feature_vector, [], 2)';% (1, 39)
            
            % Concatenate all statistics
            final_features = [feature_mean, feature_std, feature_min, feature_max];
            
            % Store features for THIS SPEAKER
            speaker_features = [speaker_features; final_features];
            speaker_labels{end+1} = speaker_name;
            
            % ALSO store in COMBINED dataset
            all_features = [all_features; final_features];
            all_labels{end+1} = speaker_name;
            
        catch ME
            fprintf('  ERROR processing %s: %s\n', audio_file, ME.message);
            continue;
        end
    end
    
    % Close progress bar
    close(h);
    
    %% ============================================
    %% SAVE INDIVIDUAL SPEAKER DATASET
    %% ============================================
    
    if ~isempty(speaker_features)
        % Create column names
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
        
        % Create table for this speaker
        if size(speaker_features, 2) == length(feature_columns)
            T_speaker = array2table(speaker_features, 'VariableNames', feature_columns);
        else
            T_speaker = array2table(speaker_features);
        end
        T_speaker.speaker = speaker_labels';
        
        % Save individual speaker file
        speaker_filename = fullfile(output_folder, sprintf('speaker_%s.xlsx', speaker_name));
        writetable(T_speaker, speaker_filename);
        
        fprintf('  ✓ Saved: speaker_%s.xlsx (%d samples)\n\n', speaker_name, size(speaker_features, 1));
    else
        fprintf('  ✗ No valid samples for %s\n\n', speaker_name);
    end
end

%% ============================================
%% CHECK IF WE HAVE DATA
%% ============================================

if isempty(all_features)
    error('No audio files were processed! Please check:\n1. Folder path is correct\n2. Audio files exist\n3. File extensions (.wav, .mp3, .FLAC, etc.)');
end

%% ============================================
%% CREATE COMBINED DATASET TABLE
%% ============================================

fprintf('\n========================================\n');
fprintf('CREATING COMBINED DATASET\n');
fprintf('========================================\n');
fprintf('Total samples: %d\n', size(all_features, 1));
fprintf('Feature dimension: %d\n', size(all_features, 2));
fprintf('Unique speakers: %d\n', length(unique(all_labels)));

% Create column names for combined dataset
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

% Create combined table
if size(all_features, 2) == length(feature_columns)
    T = array2table(all_features, 'VariableNames', feature_columns);
else
    fprintf('Warning: Column count mismatch. Using auto-generated names.\n');
    T = array2table(all_features);
end

T.speaker = all_labels';

%% ============================================
%% SAVE COMBINED DATASET
%% ============================================

fprintf('\nSaving combined dataset...\n');

% Save combined Excel
combined_xlsx = fullfile(output_folder, 'mfcc_dataset_all_speakers.xlsx');
writetable(T, combined_xlsx);
fprintf('  ✓ Saved: mfcc_dataset_all_speakers.xlsx\n');

% Save combined CSV
combined_csv = fullfile(output_folder, 'mfcc_dataset_all_speakers.csv');
writetable(T, combined_csv);
fprintf('  ✓ Saved: mfcc_dataset_all_speakers.csv\n');

% Save combined MAT file
combined_mat = fullfile(output_folder, 'mfcc_dataset_all_speakers.mat');
save(combined_mat, 'all_features', 'all_labels', 'T');
fprintf('  ✓ Saved: mfcc_dataset_all_speakers.mat\n');

%% ============================================
%% DISPLAY SUMMARY
%% ============================================

fprintf('\n========================================\n');
fprintf('DATASET SUMMARY\n');
fprintf('========================================\n');
fprintf('Combined dataset shape: %d rows × %d columns\n', size(T, 1), size(T, 2));

fprintf('\nSamples per speaker:\n');
[unique_speakers, ~, idx] = unique(T.speaker);
for i = 1:length(unique_speakers)
    count = sum(idx == i);
    fprintf('  %s: %d samples\n', unique_speakers{i}, count);
end

fprintf('\nFirst 5 rows:\n');
disp(head(T, 5));

fprintf('\n========================================\n');
fprintf('FILES CREATED\n');
fprintf('========================================\n');
fprintf('Location: %s\n\n', output_folder);

fprintf('Individual speaker files:\n');
for i = 1:length(unique_speakers)
    fprintf('  - speaker_%s.xlsx\n', unique_speakers{i});
end

fprintf('\nCombined files:\n');
fprintf('  - mfcc_dataset_all_speakers.xlsx\n');
fprintf('  - mfcc_dataset_all_speakers.csv\n');
fprintf('  - mfcc_dataset_all_speakers.mat\n');

fprintf('\n========================================\n');
fprintf('Dataset creation completed successfully!\n');
fprintf('========================================\n');

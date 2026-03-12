% Test if MATLAB can read FLAC files
test_file = 'D:\Individual projects\An-Automatic-Speaker-Recognition-System-heshan\Voices\174\174-50561-0000.FLAC';  % Pick any FLAC file
[audio, fs] = audioread(test_file);
disp('Success! MATLAB can read FLAC files');
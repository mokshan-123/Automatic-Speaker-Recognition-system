function windowed_frames = apply_window(frames)

[frame_length,num_frames] = size(frames);

% Create Hamming window
window = hamming(frame_length, 'periodic');
% window is a column vector of length frame_length

windowed_frames = frames.*window; 

%fprintf('Windowing applied:\n');
%fprintf('  Num frames:     %d\n', num_frames);
%fprintf('  Frame length:   %d\n', frame_length);
%fprintf('  Window type:    Hamming (periodic)\n');
%fprintf('  Window min:     %.4f\n', min(window));
%fprintf('  Window max:     %.4f\n', max(window));
%fprintf('  Output size:    %d × %d\n', size(windowed_frames,1), size(windowed_frames,2));

end


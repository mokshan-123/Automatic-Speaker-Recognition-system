function windowed_frames = apply_window(frames)

% frames: [frame_length × num_frames]

[frame_length, ~] = size(frames);

% Periodic Hamming window for FFT analysis
window = hamming(frame_length, 'periodic');  % column vector

% Apply window to each frame (column-wise)
windowed_frames = frames .* window;

end

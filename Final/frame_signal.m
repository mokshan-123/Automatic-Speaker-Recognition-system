function frames = frame_signal(signal, frame_length, hop_size)

if isrow(signal)
    signal = signal(:);  % ensure column vector
end

N = length(signal);

if frame_length > N
    error('Frame length cannot be larger than signal length');
end

if hop_size <= 0
    error('Hop size must be positive');
end

num_frames = floor((N - frame_length) / hop_size) + 1;

% Each column = one frame
frames = zeros(frame_length, num_frames);

for i = 1:num_frames
    start_idx = (i - 1) * hop_size + 1;
    end_idx   = start_idx + frame_length - 1;
    
    frames(:, i) = signal(start_idx : end_idx);
end

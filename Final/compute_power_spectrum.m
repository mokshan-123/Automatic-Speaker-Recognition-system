function power_spectrum = compute_power_spectrum(windowed_frames, nfft)
% windowed_frames: [frame_length × num_frames]
[frame_length, num_frames] = size(windowed_frames);

if nfft < frame_length
    error('NFFT (%d) must be >= frame_length (%d)', nfft, frame_length);
end

if ~(nfft > 0 && mod(log2(nfft), 1) == 0)
    warning('NFFT=%d is not a power of 2. FFT will be slower.', nfft);
end

num_bins = nfft/2 + 1;

% Preallocate: rows = frequency bins, columns = frames
power_spectrum = zeros(num_bins, num_frames);

% FFT along time axis (dimension 1)
fft_result = fft(windowed_frames, nfft, 1);

% Keep positive frequencies
fft_positive = fft_result(1:num_bins, :);

% Power spectrum
power_spectrum = abs(fft_positive).^2 / nfft;

end
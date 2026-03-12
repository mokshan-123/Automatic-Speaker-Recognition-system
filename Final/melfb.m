function filterbank = melfb(num_filters, nfft, fs)
% MELFB - Create mel-spaced triangular filterbank
%
% INPUTS:
%   num_filters - number of mel filters (e.g., 26)
%   nfft        - FFT size (e.g., 512)
%   fs          - sampling frequency (e.g., 16000 Hz)
%
% OUTPUT:
%   filterbank  - (num_filters × num_bins) matrix of triangular filters

% Input validation
if num_filters < 1
    error('num_filters must be >= 1');
end
if nfft < 2
    error('nfft must be >= 2');
end
if fs <= 0
    error('Sampling frequency must be positive');
end

% Frequency limits
f_min = 0;        % Start at DC (can change to 20-80 Hz to exclude low noise)
f_max = fs / 2;   % Nyquist frequency

% Convert to mel scale
mel_min = hz_to_mel(f_min);
mel_max = hz_to_mel(f_max);

% Mel-spaced points
% Create num_filters+2 points (includes left/right edges)
mel_points = linspace(mel_min, mel_max, num_filters + 2);
hz_points  = mel_to_hz(mel_points);

% Map Hz frequencies to FFT bin indices
num_bins = nfft / 2 + 1;  % Only positive frequencies

% Convert Hz to bin indices
% bin = floor(freq * nfft / fs) + 1  (MATLAB 1-indexing)
bin_points = floor(hz_points * nfft / fs) + 1;

% Clamp to valid range [1, num_bins]
bin_points = max(1, min(num_bins, bin_points));

% Check for degenerate filters (optional but recommended)
if any(diff(bin_points) == 0)
    warning('melfb:degenerateFilters', ...
            'Some filters have zero width. Consider increasing nfft or reducing num_filters.');
end

% Create triangular filterbank
filterbank = zeros(num_filters, num_bins);

for m = 1:num_filters
    left   = bin_points(m);
    center = bin_points(m + 1);
    right  = bin_points(m + 2);
    
    % Rising slope: left → center
    for k = left:center
        if center > left
            filterbank(m, k) = (k - left) / (center - left);
        end
    end
    
    % Falling slope: center → right
    for k = center:right
        if right > center
            filterbank(m, k) = (right - k) / (right - center);
        end
    end
    
    % Slaney-style normalization (normalize area to 1)
    filter_sum = sum(filterbank(m, :));
    if filter_sum > 0
        filterbank(m, :) = filterbank(m, :) / filter_sum;
    end
end

end

% Helper functions
function mel = hz_to_mel(hz)
    % O'Shaughnessy formula
    mel = 2595 * log10(1 + hz / 700);
end

function hz = mel_to_hz(mel)
    % Inverse of O'Shaughnessy formula
    hz = 700 * (10.^(mel / 2595) - 1);
end

% TECHNICAL NOTES:
% 1. Why mel scale?
%    Human perception of pitch is nonlinear. We're more sensitive to
%    differences at low frequencies than high frequencies. The mel scale
%    approximates this: 1000 mel ≈ 1000 Hz, but 2000 mel ≈ 3400 Hz.
%
% 2. Why triangular filters?
%    Triangular filters are simple, efficient, and provide smooth
%    interpolation between adjacent frequency bands. They overlap
%    so that each frequency bin contributes to at most 2 filters.
%
% 3. Filter width:
%    Filters are narrower at low frequencies (more frequency resolution
%    where humans are sensitive) and wider at high frequencies (less
%    resolution where humans are less sensitive).
%
% 4. Bin indices:
%    We map continuous Hz frequencies to discrete FFT bin indices.
%    Rounding can cause multiple filters to share the same peak bin
%    at high frequencies where filters are wide.
%
% 5. Alternative formulas:
%    Some implementations use different mel scale formulas:
%    - O'Shaughnessy: mel = 2595 * log10(1 + f/700)  [used here]
%    - Slaney (HTK): mel = 1127 * ln(1 + f/700)     [natural log]
%    The difference is just a scale factor (2595/1127 ≈ 2.303 ≈ ln(10))
%
% 6. Frequency bounds:
%    - f_min = 0 Hz: Standard choice
%    - f_min = 20-80 Hz: Sometimes used to exclude DC/low-freq noise
%    - f_max = fs/2: Nyquist (must not exceed this)
%    - f_max = 8000 Hz: Sometimes capped for telephone-quality speech
